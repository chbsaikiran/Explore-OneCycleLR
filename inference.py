import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import Net
from model import DepthwiseSeparableConv  # Import the model from model.py

# CIFAR-10 Mean and Std (calculated over the entire dataset)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# CIFAR-10 Class Labels
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Test Transformations
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
])

# Load the test dataset
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net()  # Initialize the model architecture
model = torch.load("model.pth", map_location=torch.device('cpu')) # Load the saved weights
model.to(device)
model.eval()  # Set the model to evaluation mode

# Perform inference and visualize random images
with torch.no_grad():
    # Select 10 random indices from the test dataset
    random_indices = np.random.choice(len(test_dataset), size=10, replace=False)
    fig, axs = plt.subplots(1, 10, figsize=(20, 5))

    for idx, ax in zip(random_indices, axs):
        # Get the data and target
        data, target = test_dataset[idx]
        data, target = data.to(device), torch.tensor(target).to(device)

        # Perform inference
        output = model(data.unsqueeze(0))  # Add batch dimension
        predicted = output.argmax(dim=1).item()

        # Unnormalize the image for display
        unnormalized_img = data.cpu().numpy().transpose(1, 2, 0) * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN)
        unnormalized_img = np.clip(unnormalized_img, 0, 1)  # Clip values to [0, 1]

        # Display the image
        ax.imshow(unnormalized_img)
        ax.axis("off")
        ax.set_title(f"P: {CIFAR10_CLASSES[predicted]}\nT: {CIFAR10_CLASSES[target.item()]}")

    plt.tight_layout()
    plt.show()
