from array import array
import io
import os
from pickletools import optimize
from matplotlib.pyplot import flag, imshow, plot, show
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
import random
import matplotlib.image as imgplt
import matplotlib.pyplot as plt


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

        self.fc1 = nn.Linear(512, 3)

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
        # print(out.size())
        #
        out = out.view(out.size(0), -1)
        # print(out.size())
        #ps = raw_input()
        out = torch.sigmoid(self.fc1(out))

        return out


def rightRate(numpy1, numpy2):
    f, t, rate = 0, 0, 0
    for i in range(len(numpy1)):
        if np.argmax(numpy1[i]) == np.argmax(numpy2[i]):
            t += 1
        else:
            f += 1
    rate = t/(t+f)
    return rate


def getTwoSet(sat_tiff_path, target_tif_path):
    k = 0
    pxLength = 32  # 生成总的边长为2*pxLength+1长度的正方形截图
    savePoint = []
    roadimg = io.imread(target_tif_path)
    for i in range(pxLength, 1500-pxLength-1):
        for j in range(pxLength, 1500-pxLength-1):
            if roadimg[i][j] == 255:
                savePoint.append((i,j))  # 存储路面点的坐标
                k += 1
    print('共存点数',k)
    Z = np.zeros((len(savePoint), 64, 64, 3))
    sat_image = io.imread(sat_tiff_path)
    for cut in range(len(savePoint)):
        cropped = sat_image[(savePoint[cut][0]-pxLength):(savePoint[cut][0]+pxLength),
                            (savePoint[cut][1]-pxLength):(savePoint[cut][1]+pxLength)]
        Z[cut] = cropped
    print('训练集数量', len(Z))
    return savePoint, Z


def plotByNumpy(numpy, savePoint):
    x0, x1, x2, y0, y1, y2 = [], [], [], [], [], []
    for i in range(len(numpy)):
        if np.argmax(numpy[i]) == 0:  # 土路
            x0.append(savePoint[i][0])
            y0.append(savePoint[i][1])
        elif np.argmax(numpy[i]) == 1:  # 高速路
            x1.append(savePoint[i][0])
            y1.append(savePoint[i][1])
        else:  # 社区道路
            x2.append(savePoint[i][0])
            y2.append(savePoint[i][1])
    return x0, x1, x2, y0, y1, y2


landtypes = os.listdir('images')


X = np.zeros((500, 64, 64, 3))
Y = np.zeros((500, 3))
#Z2 = np.zeros((500, 3))

idx = 0

for tidx, lt in enumerate(landtypes):
    filenames = os.listdir('images/'+lt)
    for fn in filenames:
        image = io.imread('images/'+lt+'/'+fn)
        X[idx] = image
        Y[idx, tidx] = 1
        idx += 1

savePoint, Z = getTwoSet('C:/Users/16578/Desktop/bishe/Road Detection Datasets/sat_tiff/10078675_15.tiff',
                         'C:/Users/16578/Desktop/bishe/Road Detection Datasets/target_tif/10078675_15.tif')
X = np.swapaxes(X, 1, 3)
Z = np.swapaxes(Z, 1, 3)
Z2 = np.zeros((len(Z), 3))
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.33, random_state=42)

train = data_utils.TensorDataset(torch.from_numpy(
    X_train).float(), torch.from_numpy(Y_train).float())
test = data_utils.TensorDataset(torch.from_numpy(
    X_test).float(), torch.from_numpy(Y_test).float())
train_loader = data_utils.DataLoader(train, batch_size=8, shuffle=True)
test_loader = data_utils.DataLoader(test, batch_size=165, shuffle=False)

ans = data_utils.TensorDataset(torch.from_numpy(
    Z).float(), torch.from_numpy(Z2).float())
ans_loader = data_utils.DataLoader(ans, batch_size=20, shuffle=False)

net = MyCNN()
net.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
criterion = nn.MSELoss()

optimizer = optim.SGD(net.parameters(), lr=0.01)
best_loss = float('inf')
for epoch in range(1, 16, 1):  # 16
    net.train()
    tot_loss_train = 0.0
    for (inputs, targets) in train_loader:
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
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
    print('epoch:', epoch, 'tot_loss_train:', tot_loss_train, 'tot_loss_test:' , tot_loss_test)


net.eval()

for (inputs, targets) in test_loader:
    outputs = net(inputs)
    # print(outputs.data.numpy())
    # print(targets.data.numpy())
    rate = rightRate(outputs.data.numpy(), targets.data.numpy())
    print(rate)
    # print(len(targets.data.numpy()))
    # print(np.argmax(targets.data.numpy()))
    #ps = input()
net.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
net.eval()

Z1 = np.zeros((1500, 1500, 3))
Z1[Z1 == 0] = 255
for idx,(inputs, targets) in enumerate(ans_loader):
    outputs = net(inputs)
    if idx%10 == 0:
        print('第',idx,outputs.data.numpy())
    nowpoint = savePoint[idx*len(outputs.data.numpy()) : (idx+1)*len(outputs.data.numpy())]
    x0, x1, x2, y0, y1, y2 = plotByNumpy(outputs.data.numpy(), nowpoint)
    Z1[x0, y0, 0],Z1[x0, y0, 1],Z1[x0, y0, 2] = 255,0,0     #土路red
    Z1[x1, y1, 1],Z1[x1, y1, 0],Z1[x1, y1, 2] = 255,0,0     #高速路green    
    Z1[x2, y2, 2],Z1[x2, y2, 0],Z1[x2, y2, 1] = 255,0,0     #社区道路blue
#array = []
#im = imgplt.imread('C:/Users/16578/Desktop/bishe/Road Detection Datasets/target_tif/10078675_15.tif')
plt.imshow(Z1)
plt.show()
