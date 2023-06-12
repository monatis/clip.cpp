import os
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

base_dir = os.path.dirname(__file__)
img_path = os.path.join(base_dir, "white.jpg")
img = Image.open(img_path)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
inputs = processor(images=img, return_tensors="pt")
print("white.jpg processed input sum:", inputs.pixel_values.sum())
p = inputs.pixel_values.detach().cpu().numpy().astype(np.float32)
with open(os.path.join(base_dir, "inputs.bin"), "wb") as f:
    p.tofile(f)
text_inputs = processor(text="a red apple", return_tensors="pt")
out = model.get_image_features(**inputs)
out = out.detach().squeeze().cpu().numpy()
print("white.jpg output sum:", out.sum())
out = [str(v) for v in out]


with open(os.path.join(base_dir, "white-ref.txt"), "w") as f:
    f.write("\n".join(out))

cls = (
    model.state_dict()["vision_model.embeddings.class_embedding"]
    .detach()
    .cpu()
    .numpy()
    .tolist()
)

cls = [str(v) for v in cls]

with open(os.path.join(base_dir, "cls-ref.txt"), "w") as f:
    f.write("\n".join(cls))

out = model.get_text_features(**text_inputs).detach().cpu().numpy()
print("text output sum:", out.sum())
out = [str(v) for v in out]
with open(os.path.join(base_dir, "apple-ref.txt"), "w") as f:
    f.write("\n".join(out))
