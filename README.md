# CPSC581
This README provides step-by-step instructions to reproduce the results obtained from running the code.

## Description
Before running the code, ensure that you have the following dependencies installed:

- Python (version 3.6 or above)
- PyTorch (version 1.9.0)
- Torchvision (version 0.10.0)
- Matplotlib
- Scikit-learn

It is strongly recommended to install or update the PyTorch library and the torchvision package as detailed below and in the accompanying ipynb to prevent potential errors when loading the MNIST dataset. Additionally, the data has been included in the zip file provided; therefore, if you have installed the suggested versions of torch and torchvision, the code should execute without issues.
```
!pip install --upgrade torch==1.9.0
!pip install --upgrade torchvision==0.10.0
```

The code will perform the following steps:
   - Load and preprocess the MNIST dataset.
   - Define the CNN architecture.
   - Train the CNN model using different optimizers (Adam, SGD, RMSprop) for a specified number of epochs.
   - Evaluate the model on the test set and record the accuracies for each optimizer.
   - Plot the training and validation losses for each optimizer.
   - Plot the accuracy curves for each optimizer over the epochs.

During the execution, you will see the training progress printed in the console, including the epoch number, batch number, training loss, and validation loss.

After the training is complete, the code will display the accuracies achieved by each optimizer on the test set.

Two plots will be generated and displayed:
   - "PyTorch_CNN_Loss" plot: Shows the training and validation losses for each optimizer over the batches.
   - "PyTorch_CNN_Accuracy" plot: Shows the accuracy curves for each optimizer over the epochs.

Additionally, the code will run the Random Forest and SVM classifiers from scikit-learn on the MNIST dataset and display their accuracies.

Once the execution is complete, you can review the printed accuracies and the generated plots to analyze the performance of different optimizers and compare them with the scikit-learn classifiers.

That's it! By following these steps, you should be able to reproduce the results obtained from running the code.

Please refer to the code provided in the appendix or the ipynb to replicate the results.

## Appendix
```
import torch
import torchvision
from torchvision.datasets import mnist
from torch.utils import data
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.tensor")

# Define the CNN architecture
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2*2*64, 100)
        self.mlp2 = torch.nn.Linear(100, 10)
        self.relu = torch.nn.ReLU()  

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.relu(x) 
        x = self.mlp2(x)
        return x

# Load and preprocess the dataset
def dataset(train_batch: int = 128, test_batch: int = 1000):
    data_tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
    )
    train_data = mnist.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_data = mnist.MNIST(root='./data', train=False, transform=data_tf, download=True)

    train_size = int(0.8 * len(train_data))  
    val_size = len(train_data) - train_size  
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

    train_loader = data.DataLoader(dataset=train_dataset, batch_size=train_batch, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=test_batch, shuffle=False)
    test_loader = data.DataLoader(dataset=test_data, batch_size=test_batch, shuffle=False)

    return train_data, train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_data, _, _, _ = dataset(train_batch=128, test_batch=1000)

    label_counts = {}
    for label in train_data.targets:
        if label.item() not in label_counts:
            label_counts[label.item()] = 0
        label_counts[label.item()] += 1

    for label, count in sorted(label_counts.items()):
        print(f"Number of images for class {label}: {count}")

    plt.figure(figsize=(10, 5))
    bars = plt.bar(label_counts.keys(), label_counts.values())
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Images per Class')
    plt.xticks(range(10))

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 str(int(height)), ha='center', va='bottom')

    plt.show()

# Train the CNN model
def train(train_epoch: int, train_loader, val_loader, test_loader, opt):
    model = CNNnet()
    loss_func = torch.nn.CrossEntropyLoss()

    if opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        color = 'red'
    elif opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
        color = 'green'
    elif opt == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, alpha=0.9)
        color = 'blue'
    else:
        raise ValueError("Unsupported optimizer")

    train_loss_count = []
    val_loss_count = []
    epoch_accuracies = []  # Record accuracy for each epoch

    for epoch in range(1, train_epoch+1):

        model.train()
        for i, (x, y) in enumerate(train_loader, start=1):
            out = model(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 20 == 0:
                train_loss_count.append(loss.item())

                # Evaluate the model on the validation set
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for x, y in val_loader:
                        out = model(x)
                        loss = loss_func(out, y)
                        total_val_loss += loss.item()
                val_loss_count.append(total_val_loss / len(val_loader))

                model.train()

                print('Epoch: {}, Batch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(
                    epoch, i, loss.item(), total_val_loss / len(val_loader)))

        # Evaluate the model on the test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        accuracy = 100 * correct / total
        epoch_accuracies.append(accuracy) 
        print('Accuracy on 10,000 test images: {:.2f}%'.format(accuracy))

    plt.figure('PyTorch_CNN_Loss')
    plt.plot(range(20, len(train_loss_count)*20+20, 20), train_loss_count, color=color, linestyle='-', label=f'{opt} Train Loss')
    plt.plot(range(20, len(val_loss_count)*20+20, 20), val_loss_count, color=color, linestyle='--', label=f'{opt} Validation Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()

    return epoch_accuracies 

if __name__ == "__main__":
    _, train_loader, val_loader, test_loader = dataset(train_batch=128, test_batch=1000)

    num_epochs = 10
    optimizer_accuracies = {} 

    # Run the training loop for each optimizer
    for opt in ['Adam', 'SGD', 'RMSprop']:
        print("Training using optimizer: ", opt)
        accuracies = train(train_epoch=num_epochs, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, opt=opt)
        optimizer_accuracies[opt] = accuracies

    plt.figure('PyTorch_CNN_Accuracy')
    for opt, accuracies in optimizer_accuracies.items():
        plt.plot(range(1, num_epochs+1), accuracies, marker='o', label=f'{opt} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def dataset(train_batch: int = 128, test_batch: int = 1000):
    data_tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
    )
    train_data = mnist.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_data = mnist.MNIST(root='./data', train=False, transform=data_tf, download=True)

    train_data = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    return train_data, test_data

def dataset(train_batch: int = 128, test_batch: int = 1000):
    data_tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ]
    )
    train_data = mnist.MNIST(root='./data', train=True, transform=data_tf, download=True)
    test_data = mnist.MNIST(root='./data', train=False, transform=data_tf, download=True)

    train_data = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    return train_data, test_data

```
