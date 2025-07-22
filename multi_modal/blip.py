from urllib.request import urlopen

from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch


# Prepare the model and the processor
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing
car_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png"
image = Image.open(urlopen(car_path)).convert("RGB")
inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)

text = "Her vocalization was remarkably melodic"
token_ids = blip_processor(image, text=text, return_tensors="pt").to(device, torch.float16)
print(token_ids)