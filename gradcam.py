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

device = torch.device("cpu")

model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load("/home/uif41046/Licenta/model_checkpoint.pt", map_location=device))

print(model_ft)

gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
  global gradients
  print('Backward hook running...')
  gradients = grad_output
  print(f'Gradients size: {gradients[0].size()}') 

def forward_hook(module, args, output):
  global activations
  print('Forward hook running...')
  activations = output
  print(f'Activations size: {activations.size()}')

backward_hook = model_ft.layer4[1].register_full_backward_hook(backward_hook, prepend=False)
forward_hook = model_ft.layer4[1].register_forward_hook(forward_hook, prepend=False)

img_path = "/home/uif41046/extracted_images/val/construction/{2021.06.21_at_09.45.42_camera-mi_680_mem-aff_8.rrec}_37153d9a_jpg_1624269843720123_export_odd.jpg"
image = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])
img_tensor = transform(image)

model_ft(img_tensor.unsqueeze(0)).backward(gradient=torch.ones((1,2)))

pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

for i in range(activations.size()[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

heatmap = torch.mean(activations, dim=1).squeeze()

heatmap = F.relu(heatmap)
heatmap /= torch.max(heatmap)

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(to_pil_image(img_tensor, mode='RGB'))
overlay = to_pil_image(heatmap.detach(), mode='F').resize((224,224), resample=PIL.Image.BICUBIC)
cmap = colormaps['jet']
overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
ax.imshow(overlay, alpha=0.4, interpolation='nearest')
plt.savefig("/home/uif41046/Licenta/heatmap_overlay.png")

backward_hook.remove()
forward_hook.remove()