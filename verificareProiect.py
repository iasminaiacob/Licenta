import torch
from torchvision import models, transforms
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import PIL
import cv2

device = torch.device("cpu")

model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load("/home/uif41046/Licenta/model_checkpoint.pt", map_location=device))
model_ft.eval()

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])
    img_tensor = transform(image)

    return img_tensor

image_path = input("Introduceti locatia imaginii:")
img_tensor = process_image(image_path)
#model_ft(img_tensor.unsqueeze(0)).backward(gradient=torch.ones((1,2)))

with torch.no_grad():
    outputs = model_ft(img_tensor.unsqueeze(0))
    _, preds = torch.max(outputs, 1)

class_names=["construction", "not construction"]

predicted_class = class_names[preds.item()] 

print(f"Clasa prezisa: {predicted_class}")