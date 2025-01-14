import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import WeightedRandomSampler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
import matplotlib.pyplot as plt
import time
import os
from tempfile import TemporaryDirectory
from collections import Counter
from torch.utils.tensorboard import SummaryWriter
import matplotlib.cm as cm

cudnn.benchmark = True
plt.ion() #mod interactiv

writer = SummaryWriter()

#Augmentare date pentru antrenare
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}

data_dir = "/home/uif41046/extracted_images"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
dataloaders = {}
for name, dataset in image_datasets.items():
    class_counts = dict(Counter(dataset.targets))
    weights = [2, 1]
    # sampler = WeightedRandomSampler(weights=weights, num_samples=len(dataset))
    dataloaders[name] = torch.utils.data.DataLoader(dataset, batch_size=256,
                                             shuffle=True, num_workers=16)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes #0- construction; 1- not construction


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    #Creare dosar temporar pentru a salva checkpoint-uri din antrenare
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            #Fiecare epoca are o faza de antrenare si una de validare
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train() #Modul antrenare
                else:
                    model.eval() #Modul evaluare

                running_loss = 0.0
                running_corrects = 0

                tps = 0
                fps = 0
                fns = 0

                #Iterarea datelor
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    #Gradientii parametrilor se seteaza cu 0
                    optimizer.zero_grad()

                    #forward
                    #Monitorizare istoric daca modelul este in modul de antrenare
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        #backward
                        #Optimizare daca modelul este in modul de evaluare
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        if phase == 'val':
                            y_true = labels.data
                            y_pred = preds

                            tps += (y_true * y_pred).sum().to(torch.float32)
                            fps += ((1 - y_true) * y_pred).sum().to(torch.float32)
                            fns += (y_true * (1 - y_pred)).sum().to(torch.float32)
                            
                    #Statistici
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                epsilon = 1e-7
                precision = tps / (tps + fps + epsilon)
                recall = tps / (tps + fns + epsilon)
                f1 = 2* (precision*recall) / (precision + recall + epsilon)
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    print(f'{phase} Eval_f1: {f1:.4f} Eval_precision: {precision:.4f} Eval_recall: {recall:.4f}')
                    writer.add_scalar("Evaluate f1", f1, epoch)
                    writer.add_scalar("Evaluate precision", precision, epoch)
                    writer.add_scalar("Evaluate recall", recall, epoch)
                    writer.add_scalar("Evaluate loss", epoch_loss, epoch)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path))
    return model

def visualize_model(model, num_images):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(1, 1, 1)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}, label: {class_names[labels[j]]}')
                ax.imshow(inputs.cpu().data[j].permute(1, 2, 0))
                plt.savefig(f"/home/uif41046/Licenta/eval_imgs/{j}_eval_{class_names[preds[j]]}_{class_names[labels[j]]}.png")

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

weight_tensor= torch.tensor([10.0, 2.0], device=device)
criterion = nn.CrossEntropyLoss(weight=weight_tensor)

optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.7)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=30)

torch.save(model_ft.state_dict(), "/home/uif41046/Licenta/model_checkpoint.pt")

writer.flush()
writer.close()

visualize_model(model_ft, 100)