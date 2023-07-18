import ctypes
import os

# Load the shared library
path_to_dll = os.environ.get("CLIP_DLL", "./libclip.so")

clip_lib = ctypes.CDLL(path_to_dll)


# Define the ctypes structures
class ClipTextHparams(ctypes.Structure):
    _fields_ = [
        ("n_vocab", ctypes.c_int32),
        ("num_positions", ctypes.c_int32),
        ("hidden_size", ctypes.c_int32),
        ("n_intermediate", ctypes.c_int32),
        ("projection_dim", ctypes.c_int32),
        ("n_head", ctypes.c_int32),
        ("n_layer", ctypes.c_int32),
    ]


class ClipVisionHparams(ctypes.Structure):
    _fields_ = [
        ("image_size", ctypes.c_int32),
        ("patch_size", ctypes.c_int32),
        ("hidden_size", ctypes.c_int32),
        ("n_intermediate", ctypes.c_int32),
        ("projection_dim", ctypes.c_int32),
        ("n_head", ctypes.c_int32),
        ("n_layer", ctypes.c_int32),
    ]


class ClipVocabId(ctypes.c_int32):
    pass


class ClipVocabToken(ctypes.c_char_p):
    pass


class ClipVocabSpecialTokens(ctypes.c_char_p):
    pass


class ClipVocab(ctypes.Structure):
    _fields_ = [
        ("token_to_id", ctypes.POINTER(ctypes.c_void_p)),
        ("id_to_token", ctypes.POINTER(ctypes.c_void_p)),
        ("special_tokens", ctypes.POINTER(ClipVocabSpecialTokens)),
    ]


class ClipLayer(ctypes.Structure):
    _fields_ = [
        ("k_w", ctypes.POINTER(ctypes.c_void_p)),
        ("k_b", ctypes.POINTER(ctypes.c_void_p)),
        ("q_w", ctypes.POINTER(ctypes.c_void_p)),
        ("q_b", ctypes.POINTER(ctypes.c_void_p)),
        ("v_w", ctypes.POINTER(ctypes.c_void_p)),
        ("v_b", ctypes.POINTER(ctypes.c_void_p)),
        ("o_w", ctypes.POINTER(ctypes.c_void_p)),
        ("o_b", ctypes.POINTER(ctypes.c_void_p)),
        ("ln_1_w", ctypes.POINTER(ctypes.c_void_p)),
        ("ln_1_b", ctypes.POINTER(ctypes.c_void_p)),
        ("ff_i_w", ctypes.POINTER(ctypes.c_void_p)),
        ("ff_i_b", ctypes.POINTER(ctypes.c_void_p)),
        ("ff_o_w", ctypes.POINTER(ctypes.c_void_p)),
        ("ff_o_b", ctypes.POINTER(ctypes.c_void_p)),
        ("ln_2_w", ctypes.POINTER(ctypes.c_void_p)),
        ("ln_2_b", ctypes.POINTER(ctypes.c_void_p)),
    ]


class ClipTextModel(ctypes.Structure):
    _fields_ = [
        ("hparams", ClipTextHparams),
        ("token_embeddings", ctypes.POINTER(ctypes.c_void_p)),
        ("position_embeddings", ctypes.POINTER(ctypes.c_void_p)),
        ("layers", ctypes.POINTER(ClipLayer)),
        ("post_ln_w", ctypes.POINTER(ctypes.c_void_p)),
        ("post_ln_b", ctypes.POINTER(ctypes.c_void_p)),
        ("projection", ctypes.POINTER(ctypes.c_void_p)),
        ("tensors", ctypes.POINTER(ctypes.c_void_p)),
    ]


class ClipVisionModel(ctypes.Structure):
    _fields_ = [
        ("hparams", ClipVisionHparams),
        ("class_embedding", ctypes.POINTER(ctypes.c_void_p)),
        ("patch_embeddings", ctypes.POINTER(ctypes.c_void_p)),
        ("position_embeddings", ctypes.POINTER(ctypes.c_void_p)),
        ("pre_ln_w", ctypes.POINTER(ctypes.c_void_p)),
        ("pre_ln_b", ctypes.POINTER(ctypes.c_void_p)),
        ("layers", ctypes.POINTER(ClipLayer)),
        ("post_ln_w", ctypes.POINTER(ctypes.c_void_p)),
        ("post_ln_b", ctypes.POINTER(ctypes.c_void_p)),
        ("projection", ctypes.POINTER(ctypes.c_void_p)),
        ("tensors", ctypes.POINTER(ctypes.c_void_p)),
    ]


class ClipBuffer(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("size", ctypes.c_size_t),
    ]


class ClipImageU8(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


class ClipImageF32(ctypes.Structure):
    _fields_ = [
        ("nx", ctypes.c_int),
        ("ny", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_float)),
    ]


# Load the functions from the shared library
clip_model_load = clip_lib.clip_model_load
clip_model_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
clip_model_load.restype = ctypes.POINTER(ctypes.c_void_p)

clip_free = clip_lib.clip_free
clip_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

clip_tokenize = clip_lib.clip_tokenize
clip_tokenize.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
clip_tokenize.restype = ctypes.POINTER(ClipVocabId)

clip_image_load_from_file = clip_lib.clip_image_load_from_file
clip_image_load_from_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ClipImageU8)]
clip_image_load_from_file.restype = ctypes.c_bool

clip_image_preprocess = clip_lib.clip_image_preprocess
clip_image_preprocess.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ClipImageU8),
    ctypes.POINTER(ClipImageF32),
]
clip_image_preprocess.restype = ctypes.c_bool

clip_text_encode = clip_lib.clip_text_encode
clip_text_encode.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_int,
    ctypes.POINTER(ClipVocabId),
    ctypes.POINTER(ctypes.c_float),
]
clip_text_encode.restype = ctypes.c_bool

clip_image_encode = clip_lib.clip_image_encode
clip_image_encode.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_int,
    ctypes.POINTER(ClipImageF32),
    ctypes.POINTER(ctypes.c_float),
]
clip_image_encode.restype = ctypes.c_bool

clip_compare_text_and_image = clip_lib.clip_compare_text_and_image
clip_compare_text_and_image.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_int,
    ctypes.c_char_p,
    ctypes.POINTER(ClipImageU8),
    ctypes.POINTER(ctypes.c_float),
]
clip_compare_text_and_image.restype = ctypes.c_bool

clip_similarity_score = clip_lib.clip_similarity_score
clip_similarity_score.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
]
clip_similarity_score.restype = ctypes.c_float

softmax_with_sorting = clip_lib.softmax_with_sorting
softmax_with_sorting.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_int),
]
softmax_with_sorting.restype = ctypes.c_bool

clip_image_batch_encode = clip_lib.clip_image_batch_encode
clip_image_batch_encode.argtypes = [
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.c_int,
    ctypes.POINTER(ClipImageF32),
    ctypes.POINTER(ctypes.c_float),
]
clip_image_batch_encode.restype = ctypes.c_bool


class Clip:
    def __init__(self, model_file: str, verbose=0):
        self.ctx = clip_model_load(model_file.encode("utf8"), verbose)

    def __del__(self):
        clip_free(self.ctx)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="clip")
    ap.add_argument("-m", "--model", help="path to GGML file")
    args = ap.parse_args()

    clip = Clip(args.model, 2)
