from __future__ import print_function
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from model import Net  # Import the model from model.py

# CIFAR-10 Mean and Std (calculated over the entire dataset)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Albumentations Transformations
class AlbumentationsTransform:
    def __init__(self, mean, std):
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),  # Random horizontal flip
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),  # Shift, Scale, Rotate
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, 
                            fill_value=(np.array(mean) * 255).tolist(), mask_fill_value=None, p=0.5),  # CoarseDropout
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),  # Normalize
            ToTensorV2(),  # Convert to tensor
        ])
    
    def __call__(self, img):
        # Albumentations expects the image in OpenCV format (HWC, uint8)
        image = np.array(img)  # Convert PIL image to numpy array
        return self.transform(image=image)["image"]

# Apply Albumentations Transformations to CIFAR-10
train = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=AlbumentationsTransform(mean=CIFAR10_MEAN, std=CIFAR10_STD)
)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

test = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)


from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
model = Net().to(device)
summary(model, input_size=(3, 32, 32))

from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

criterion = nn.CrossEntropyLoss()
model =  Net().to(device)

def train(model, device, train_loader, optimizer,scheduler, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))


# LR Finder
def find_lr(model, train_loader, criterion, optimizer, device):
    model.train()
    lrs = []
    losses = []
    
    min_lr = 1e-6
    max_lr = 1
    num_steps = len(train_loader)
    print(f"Number of steps: {num_steps}")
    
    # Corrected lambda function for learning rate scaling
    lr_lambda = lambda x: min_lr * (max_lr / min_lr) ** (x / num_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for i, (inputs, labels) in enumerate(train_loader):
        if i >= num_steps:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        lrs.append(scheduler.get_last_lr()[0])
        losses.append(loss.item())

    # Plot the learning rate vs. loss
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('LR Finder')
    plt.show()

    return lrs, losses

optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9)
print("Finding optimal LR...")
lrs, losses = find_lr(model, train_loader, criterion, optimizer, device)
best_lr = lrs[losses.index(min(losses))]
print(f"Optimal LR: {best_lr}")


from torch.optim.lr_scheduler import OneCycleLR

EPOCHS = 50
best_lr = 0.0004
optimizer = optim.SGD(model.parameters(), lr=best_lr, momentum=0.9)
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.1,  # max_lr provided to OneCycleLR
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    div_factor=(0.1/best_lr),   # Determines starting LR
    final_div_factor=1e4
)

for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer,scheduler, epoch)
    test(model, device, test_loader)
    print(f"Learning Rate = {optimizer.param_groups[0]['lr']}\n")
    if (test_acc[-1] > 86):
      break