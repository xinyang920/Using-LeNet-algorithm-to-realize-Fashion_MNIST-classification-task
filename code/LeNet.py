import torch
import torchvision
from torch import nn
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np 


## Define a transform to read the data in as a tensor
data_transform = transforms.ToTensor()

# choose the training and test datasets
minist_train = FashionMNIST(root='./data',train=True,download=True,transform=transforms.ToTensor())
minist_test = FashionMNIST(root='./data',train=False,download=True,transform=transforms.ToTensor())

# Print out some stats about the training data
print('Train data, number of images: ', len(train_data))

batch_size = 32 
train_loader = DataLoader(minist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(minist_test, batch_size=batch_size, shuffle=False)
# specify the image classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# define LeNet
class myLeNet(nn.Module):
    def __init__(self):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=5)
        self.pooling = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size,-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
net = myLeNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
# start training
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader,0):
        inputs,target = data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs,target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("[epoch => %d,batch_idx => %5d] loss:%.3f" %(epoch+1,batch_idx+1,running_loss/2000))
            running_loss = 0.0

def test(data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            inputs,target = data
            inputs,target = inputs.to(device),target.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target ).sum().item()
    return 100*correct/total

def Accuracy_plot():
    plt.figure()
    plt.plot(range(10,10*len(train_acc)+1,10),train_acc,c="r",label='train')
    plt.plot(range(10,10*len(train_acc)+1,10),test_acc,c="b", label='test')
    plt.legend()
    plt.title("Accuracy of training and testing")
    plt.show()


train_acc = []
test_acc = []
for epoch in range(1,11):
    train(epoch)
    # if epoch%10 == 0:
    train_acc.append(test(train_loader))
    test_acc.append(test(test_loader))
Accuracy_plot()
