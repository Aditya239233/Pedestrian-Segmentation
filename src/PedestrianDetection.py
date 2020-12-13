#%%
from dataset import Dataset
from model import model
from helper import add_pallet, collate_fn, get_transform

import torch
from PIL import Image
from matplotlib import pyplot as plt
from engine import train_one_epoch, evaluate

#%%
image = Image.open('../data/PNGImages/FudanPed00001.png')
mask = Image.open('../data/PedMasks/FudanPed00001_mask.png')
_, axs = plt.subplots(1, 2, figsize=(12, 12))
axs = axs.flatten()
imgs = [image, mask]
for img, ax in zip(imgs, axs):
    ax.imshow(img)
plt.show()

#%%
dataset_train = Dataset('../data/', get_transform(train=True))
dataset_test = Dataset('../data/', get_transform(train=False))

print(dataset_train[0])

#%%
torch.manual_seed(1)
indices = torch.randperm(len(dataset_train)).tolist()
dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=collate_fn)

#%%
device = torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

model = model(num_classes)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

#%%
num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    lr_scheduler.step()

    evaluate(model, data_loader_test, device=device)

torch.save(model.state_dict(), "../model/model.pth")

#%%
img, _ = dataset_test[0]

model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])
print(prediction)
#%%
image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
_, axs = plt.subplots(1, 2, figsize=(12, 12))
axs = axs.flatten()
imgs = [image, mask]
for img, ax in zip(imgs, axs):
    ax.imshow(img)
plt.show()