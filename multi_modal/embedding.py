from urllib.request import urlopen
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPTokenizerFast, CLIPModel, CLIPProcessor


puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
image = Image.open(urlopen(puppy_path)).convert('RGB')
caption = "a puppy playing in the snow"

model_id = "openai/clip-vit-base-patch32"
clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
clip_processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

inputs = clip_tokenizer(caption, return_tensors="pt")
text_embedding = model.get_text_features(inputs["input_ids"])

processed_image = clip_processor(
    text=None, images=image, return_tensors="pt"
)["pixel_values"]
image_embedding = model.get_image_features(processed_image)

# print(image_embedding.shape)
img = processed_image.squeeze(0)
img = img.permute(*torch.arange(img.ndim - 1, -1, -1))
img = np.einsum("ijk->jik", img)
# Visualize preprocessed image
plt.imshow(img)
plt.axis("off")
plt.show()

text_embedding = text_embedding.norm(dim=-1, keepdim=True)
image_embedding = image_embedding.norm(dim=-1, keepdim=True)

text_embedding = text_embedding.detach().cpu().numpy()
image_embedding = image_embedding.detach().cpu().numpy()

score = np.dot(text_embedding, image_embedding.T)
print(score)

