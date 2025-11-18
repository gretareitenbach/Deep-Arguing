import torch
import torchvision
import torchvision.transforms as transforms
from torch.profiler import record_function
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)


# ----- same transform as before -----
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 500

# -------------------------------------------------------
# Load training set normally (CPU) so transforms still work
# -------------------------------------------------------
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# Preload entire dataset onto GPU
train_data = torch.stack([trainset[i][0] for i in range(len(trainset))]).cuda()
train_targets = torch.tensor([trainset[i][1] for i in range(len(trainset))]).cuda()

# Wrap GPU tensors in a Dataset class
class GPUCIFAR10(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def __len__(self):
        return self.data.size(0)

# Replace trainset with the GPU version
trainset = GPUCIFAR10(train_data, train_targets)

# IMPORTANT: num_workers must be 0 (workers cannot share GPU tensors)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=0
)

# -------------------------------------------------------
# Same process for test set
# -------------------------------------------------------
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

test_data = torch.stack([testset[i][0] for i in range(len(testset))]).cuda()
test_targets = torch.tensor([testset[i][1] for i in range(len(testset))]).cuda()

testset = GPUCIFAR10(test_data, test_targets)

testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=0
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    #    def __init__(self):
    #        super().__init__()
    #        self.conv1 = nn.Conv2d(3, 6, 5)
    #        self.pool = nn.MaxPool2d(2, 2)
    #        self.conv2 = nn.Conv2d(6, 16, 5)
    #        self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #        self.fc2 = nn.Linear(120, 84)
    #        self.fc3 = nn.Linear(84, 10)
    #
    #    def forward(self, x):
    #        x = self.pool(F.relu(self.conv1(x)))
    #        x = self.pool(F.relu(self.conv2(x)))
    #        x = torch.flatten(x, 1) # flatten all dimensions except batch
    #        x = F.relu(self.fc1(x))
    #        x = F.relu(self.fc2(x))
    #        x = self.fc3(x)
    #        return x
    def __init__(self, num_classes=10):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32 * 2, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32 * 2, out_channels=64 * 2, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64 * 2, out_channels=64 * 2, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=64 * 2, out_channels=128 * 2, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=128 * 2, out_channels=256 * 2, kernel_size=3, padding=1
        )
        self.conv8 = nn.Conv2d(
            in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1
        )
        self.conv9 = nn.Conv2d(
            in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1
        )

        self.bn1 = nn.BatchNorm2d(32 * 2)
        self.bn2 = nn.BatchNorm2d(128 * 2)
        self.bn3 = nn.BatchNorm2d(256 * 2)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(4096 * 2, 4096 * 2)
        self.fc2 = nn.Linear(4096 * 2, 2048 * 2)
        self.fc3 = nn.Linear(2048 * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv7(x)))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


net = Net().to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.AdamW(net.parameters(), lr=0.001)

epochs = 1

pbar = tqdm(range(epochs), dynamic_ncols=True)
with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler_logs_1"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # since with_stack=True caused segfault
) as prof:
    for epoch in pbar:  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            with record_function("train_step"):
                with record_function("my_move_data"):
                    inputs, labels = data
                    # inputs = inputs.to(device)
                    # labels = labels.to(device)
    
                # zero the parameter gradients
                with record_function("my_zero_grad"):
                    optimizer.zero_grad()
    
                # forward + backward + optimize
                with record_function("my_forward"):
                    outputs = net(inputs)
                with record_function("my_compute_loss"):
                    loss = criterion(outputs, labels)
                with record_function("my_backward"):
                    loss.backward()
                with record_function("my_step"):
                    optimizer.step()
    
            # print statistics
            # running_loss += loss.item()
            # pbar.set_description(f"Epoch {epoch + 1}, Loss: {round(loss.item(), 6)}")

        prof.step()
print("Finished Training")

dataiter = iter(testloader)
images, labels = next(dataiter)

images = images.to(device)
labels = labels.to(device)

# print images
# imshow(torchvision.utils.make_grid(images))
print("GroundTruth: ", " ".join(f"{classes[labels[j]]:5s}" for j in range(4)))

outputs = net(images)
_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join(f"{classes[predicted[j]]:5s}" for j in range(4)))
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the network on the 10000 test images: {100 * correct // total} %")
# prepare to count predictions for each class
