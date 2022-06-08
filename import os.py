import io
import os
from pickletools import optimize
import skimage
from matplotlib import image
from skimage import io
from skimage.transform import resize
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=5, padding=2)

        self.max_pool8 = torch.nn.MaxPool2d(8, 8)
        
        self.fc1 = nn.Linear(64, 21)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.max_pool8(out)
        out = out.view(out.size(0), -1)
        #print(out.size())
        #ps = raw_input()
        out = torch.sigmoid(self.fc1(out))

        return out'''

class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(128)
        self.max_pool2 = torch.nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512, 21)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.max_pool2(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.max_pool2(out)
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = self.max_pool2(out)
        out = F.relu(self.bn7(self.conv7(out)))
        out = F.relu(self.bn8(self.conv8(out)))
        out = self.max_pool2(out)
        out = F.relu(self.bn9(self.conv9(out)))
        out = F.relu(self.bn10(self.conv10(out)))
        out = self.max_pool2(out)
        #print(out.size())
        #
        out = out.view(out.size(0), -1)
        #print(out.size())
        #ps = raw_input()
        out = torch.sigmoid(self.fc1(out))

        return out

landtypes = os.listdir('images')

X = np.zeros((2100,64,64,3))
Y = np.zeros((2100,21))
idx = 0

for tidx,lt in enumerate(landtypes):
    filenames = os.listdir('images/'+lt)
    #print(filenames)
    for fn in filenames:
        image = io.imread('images/'+lt+'/'+fn)
        resized_image = resize(image,(64,64))
        X[idx] = resized_image
        Y[idx,tidx] = 1
        idx += 1

X = np.swapaxes(X,1,3)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

train = data_utils.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float())
test = data_utils.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y_test).float())
train_loader = data_utils.DataLoader(train, batch_size=8, shuffle=True)
test_loader = data_utils.DataLoader(test, batch_size=4, shuffle=False)

net = MyCNN()

criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(),lr=0.01)

for epoch in range(1,2,1):
    net.train()
    tot_loss_train = 0.0
    for (inputs, targets) in train_loader:
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tot_loss_train += loss.item()

    net.eval()
    tot_loss_test = 0.0
    for (inputs, targets) in test_loader: 
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        tot_loss_test += loss.item()
    print(epoch,tot_loss_train,tot_loss_test)                                                                                                                                                                                                    
    

net.eval()

for (inputs, targets) in test_loader: 
    outputs = net(inputs)
    print(outputs.data.numpy())
    print(targets.data.numpy())
    ps = input()
