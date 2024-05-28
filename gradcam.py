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
#Marimea fiecarei mostre de iesire se seteaza la 2
#In mod alternativ, poate fi generalizata astfel ``nn.Linear(num_ftrs, len(class_names))``
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft.load_state_dict(torch.load("/home/uif41046/Licenta/model_checkpoint.pt", map_location=device))

print(model_ft)

# defines two global scope variables to store our gradients and activations
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
  global gradients # refers to the variable in the global scope
  print('Backward hook running...')
  gradients = grad_output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Gradients size: {gradients[0].size()}') 
  # We need the 0 index because the tensor containing the gradients comes
  # inside a one element tuple.

def forward_hook(module, args, output):
  global activations # refers to the variable in the global scope
  print('Forward hook running...')
  activations = output
  # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
  print(f'Activations size: {activations.size()}')

backward_hook = model_ft.layer4[1].register_full_backward_hook(backward_hook, prepend=False)
forward_hook = model_ft.layer4[1].register_forward_hook(forward_hook, prepend=False)

img_path = "/home/uif41046/extracted_images/val/construction/{2021.06.21_at_09.15.39_camera-mi_680_mem-aff_7.rrec}_37153d9a_jpg_1624268347624422_export_odd.jpg"
image = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
        ])
img_tensor = transform(image)

model_ft(img_tensor.unsqueeze(0)).backward(gradient=torch.ones((1,2)))

# pool the gradients across the channels
pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

# weight the channels by corresponding gradients
for i in range(activations.size()[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
heatmap = F.relu(heatmap)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# Create a figure and plot the first image
fig, ax = plt.subplots()
ax.axis('off') # removes the axis markers

# First plot the original image
ax.imshow(to_pil_image(img_tensor, mode='RGB'))

# Resize the heatmap to the same size as the input image and defines
# a resample algorithm for increasing image resolution
# we need heatmap.detach() because it can't be converted to numpy array while
# requiring gradients
overlay = to_pil_image(heatmap.detach(), mode='F').resize((224,224), resample=PIL.Image.BICUBIC)

# Apply any colormap you want
cmap = colormaps['jet']
overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

# Plot the heatmap on the same axes, 
# but with alpha < 1 (this defines the transparency of the heatmap)

ax.imshow(overlay, alpha=0.4, interpolation='nearest')

plt.savefig("/home/uif41046/Licenta/heatmap_overlay.png")

backward_hook.remove()
forward_hook.remove()