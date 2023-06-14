import sys
import struct
import json
import torch
import numpy as np

from transformers import CLIPModel


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


if len(sys.argv) < 3:
    print("Usage: convert-h5-to-ggml.py dir-model ftype\n")
    print("  ftype == 0 -> float32")
    print("  ftype == 1 -> float16")
    sys.exit(1)

# output in the same directory as the model
dir_model = sys.argv[1]


with open(dir_model + "/vocab.json", "r", encoding="utf-8") as f:
    encoder = json.load(f)

with open(dir_model + "/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
    v_hparams = config["vision_config"]
    t_hparams = config["text_config"]

# possible data types
#   ftype == 0 -> float32
#   ftype == 1 -> float16
#
# map from ftype to string
ftype_str = ["f32", "f16"]

ftype = 1
if len(sys.argv) > 2:
    ftype = int(sys.argv[2])
    if ftype < 0 or ftype > 1:
        print("Invalid ftype: " + str(ftype))
        sys.exit(1)
    fname_out = sys.argv[1] + "/ggml-model-" + ftype_str[ftype] + ".bin"


model = CLIPModel.from_pretrained(dir_model)
# print (model)

list_vars = model.state_dict()
# print (list_vars)

fout = open(fname_out, "wb")

fout.write(struct.pack("i", 0x67676D6C))  # magic: ggml in hex

# text_model hparams
fout.write(struct.pack("i", t_hparams["vocab_size"]))
fout.write(struct.pack("i", t_hparams["max_position_embeddings"]))
fout.write(struct.pack("i", t_hparams["hidden_size"]))
fout.write(struct.pack("i", t_hparams["intermediate_size"]))
fout.write(struct.pack("i", t_hparams["projection_dim"]))
fout.write(struct.pack("i", t_hparams["num_attention_heads"]))
fout.write(struct.pack("i", t_hparams["num_hidden_layers"]))

# vision_model hparams
fout.write(struct.pack("i", v_hparams["image_size"]))
fout.write(struct.pack("i", v_hparams["patch_size"]))
fout.write(struct.pack("i", v_hparams["hidden_size"]))
fout.write(struct.pack("i", v_hparams["intermediate_size"]))
fout.write(struct.pack("i", v_hparams["projection_dim"]))
fout.write(struct.pack("i", v_hparams["num_attention_heads"]))
fout.write(struct.pack("i", v_hparams["num_hidden_layers"]))
fout.write(struct.pack("i", ftype))


byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}
fout.write(struct.pack("i", len(encoder)))

for key in encoder:
    text = bytearray([byte_decoder[c] for c in key])
    fout.write(struct.pack("i", len(text)))
    fout.write(text)


for name in list_vars.keys():
    if name in [
        "logit_scale",
        "text_model.embeddings.position_ids",
        "vision_model.embeddings.position_ids",
    ]:
        # we don't need this
        print(f"skipping parameter: {name}")
        continue

    data = list_vars[name].squeeze().numpy()
    print("Processing variable: " + name + " with shape: ", data.shape)

    n_dims = len(data.shape)

    # ftype == 0 -> float32, ftype == 1 -> float16
    ftype_cur = 0
    if ftype != 0:
        if name[-7:] == ".weight" and n_dims >= 2:
            print("  Converting to float16")
            data = data.astype(np.float16)
            ftype_cur = 1
        else:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0
    else:
        if data.dtype != np.float32:
            print("  Converting to float32")
            data = data.astype(np.float32)
            ftype_cur = 0

    # header
    str = name.encode("utf-8")
    fout.write(struct.pack("iii", n_dims, len(str), ftype_cur))
    for i in range(n_dims):
        fout.write(struct.pack("i", data.shape[n_dims - 1 - i]))
    fout.write(str)

    # data
    data.tofile(fout)

fout.close()

print("Done. Output file: " + fname_out)
print("")
