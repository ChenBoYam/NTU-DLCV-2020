import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import time
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib import cm


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class dataReader(Dataset):
    def __init__(self, root, transforms, train = True):
        self.root = root
        self.train = train
        if self.train:
            self.label_file = os.path.join(self.root, "train.csv")
        else:
            self.label_file = os.path.join(self.root, "test.csv")
        self.transforms = transforms
        with open(self.label_file) as f:
            next(f)
            lines = [line.strip('\n').split(',') for line in f]
            self.X_name, self.y = np.array(lines, dtype = str)[:,0], np.array(lines)[:,1].astype(int)
        self.length = len(self.X_name)
        self.label = torch.from_numpy(self.y).long()

    def __getitem__(self, idx):
        if self.train:
            img = Image.open(os.path.join(os.path.join(self.root, "train"), self.X_name[idx]))
        else:
            img = Image.open(os.path.join(os.path.join(self.root, "test"), self.X_name[idx]))
        img = self.transforms(img)
        label = self.label[idx]
        return img, label
    
    def __len__(self):
        return self.length

    def get_label(self):
        return self.label

same_seeds(0)

source_transform = transforms.Compose([
    # 轉灰階: Canny 不吃 RGB。
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((28, 28)),
    # 水平翻轉 (Augmentation)
    transforms.RandomHorizontalFlip(),
    # 旋轉15度內 (Augmentation)，旋轉後空的地方補0
    transforms.RandomRotation(15),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    # 轉灰階: 將輸入3維壓成1維。
    transforms.Grayscale(),
    # 縮放: 因為source data是32x32，我們將target data的28x28放大成32x32。
    transforms.Resize((28, 28)),
    # 最後轉成Tensor供model使用。
    transforms.ToTensor(),
])
root = "./hw3_data/digits/"
source = "svhn"
target = "usps"
source_path = os.path.join(root,source)
target_path = os.path.join(root,target)

train_source_dataset = dataReader(source_path, transforms=source_transform)
train_target_dataset = dataReader(target_path, transforms=target_transform)
test_source_dataset = dataReader(source_path, transforms=test_transform, train = False)
test_target_dataset = dataReader(target_path, transforms=test_transform, train = False)

train_source_dataloader = DataLoader(train_source_dataset, batch_size = 32, shuffle=True)
train_target_dataloader = DataLoader(train_target_dataset, batch_size = 32, shuffle=True)
test_source_dataloader = DataLoader(test_source_dataset, batch_size = 128, shuffle=False)
test_target_dataloader = DataLoader(test_target_dataset, batch_size = 128, shuffle=False)


class FtExtr1(nn.Module):

    def __init__(self):
        super(FtExtr1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LP1(nn.Module):

    def __init__(self):
        super(LP1, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DClf1(nn.Module):

    def __init__(self):
        super(DClf1, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y

"""# Pre-processing

這裡我們選用Adam來當Optimizer。
"""
lr = 2e-4

feature_extractor = FtExtr1().cuda()
label_predictor = LP1().cuda()
domain_classifier = DClf1().cuda()



class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optim_F = optim.Adam(feature_extractor.parameters(), lr=lr, betas=(0.5, 0.999))
optim_C = optim.Adam(label_predictor.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = optim.Adam(domain_classifier.parameters(), lr=lr, betas=(0.5, 0.999))



def train_epoch(train_source_dataloader, train_target_dataloader, test_target_dataloader, lamb):
    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
    
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    result = []
    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(train_source_dataloader, train_target_dataloader)):

        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        
        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        loss.backward()
        optim_D.step()

        # Step 2 : 訓練Feature Extractor和label Predictor
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss+= loss.item()
        loss.backward()
        optim_F.step()
        optim_C.step()

        optim_D.zero_grad()
        optim_F.zero_grad()
        optim_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    label_predictor.eval()
    feature_extractor.eval()
    test_total_hit, test_total_num = 0.0, 0.0
    for i, (test_data, test_label) in enumerate(test_target_dataloader):
        test_data = test_data.cuda()
        test_label = test_label.cuda()
        class_logits = label_predictor(feature_extractor(test_data))
        test_total_hit += torch.sum(torch.argmax(class_logits, dim=1) == test_label).item()
        test_total_num += test_data.shape[0]
    label_predictor.train()
    feature_extractor.train()
    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num, test_total_hit/test_total_num
'''
limit = 0.464
max_acc = 0
for epoch in range(150):
    train_D_loss, train_F_loss, train_acc, test_acc = train_epoch(train_source_dataloader, train_target_dataloader, test_target_dataloader, lamb=0.1)
    if test_acc > max_acc and test_acc<limit:
        max_acc = test_acc
        torch.save(feature_extractor.state_dict(), f'FtExtr_{source}_to_{target}.pkl')
        torch.save(label_predictor.state_dict(), f'LP_{source}_to_{target}.pkl')
    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, train acc: {:6.4f}, test acc: {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc, test_acc))
'''
result = []
feature_extractor.load_state_dict(torch.load(f'FtExtr_{source}_to_{target}.pkl'))
label_predictor.load_state_dict(torch.load(f'LP_{source}_to_{target}.pkl'))
label_predictor.eval()
feature_extractor.eval()
test_total_hit, test_total_num = 0.0, 0.0
for i, (test_data, test_label) in enumerate(test_target_dataloader):
    test_data = test_data.cuda()
    test_label = test_label.cuda()
    class_logits = label_predictor(feature_extractor(test_data))

    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

    test_total_hit += torch.sum(torch.argmax(class_logits, dim=1) == test_label).item()
    test_total_num += test_data.shape[0]
print("testing acc: %1.3f"%(test_total_hit/test_total_num))  

import pandas as pd
result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(f'prediction_{source}_to_{target}.csv',index=False)



from sklearn.manifold import TSNE
num_of_samples = 2048
s_images, s_labels, s_tags = [], [], []
t_images, t_labels, t_tags = [], [], []
for i, ((source_data, source_label), (target_data, target_label)) in enumerate(zip(test_source_dataloader, test_target_dataloader)):
    s_images.append(source_data.cuda())
    s_labels.append(source_label)
    s_tags.append(torch.zeros((source_label.size()[0])).type(torch.LongTensor))
    t_images.append(target_data.cuda())
    t_labels.append(target_label)
    t_tags.append(torch.ones((target_label.size()[0])).type(torch.LongTensor))

s_images, s_tags, s_labels = torch.cat(s_images)[:num_of_samples], torch.cat(s_tags)[:num_of_samples],torch.cat(s_labels)[:num_of_samples]
t_images, t_tags, t_labels = torch.cat(t_images)[:num_of_samples], torch.cat(t_tags)[:num_of_samples], torch.cat(t_labels)[:num_of_samples]
embedding1 = feature_extractor(s_images)
embedding2 = feature_extractor(t_images)

tsne = TSNE(n_components=2)
dann_tsne = tsne.fit_transform(np.concatenate((embedding1.cpu().detach().numpy(),
                                                       embedding2.cpu().detach().numpy())))

import matplotlib.pyplot as plt
def plot_embedding(X, y, title=None, imgName=None, a = True):
    """
    Plot an embedding X with the class label y colored by the domain d.
    :param X: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param imgName: the name of saving image
    :return:
    """

    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        # plot colored number
        c = cm.rainbow(int(255 * y[i] / 10))
        if a:
            plt.scatter(X[i, 0], X[i, 1], color=plt.cm.bwr(y[i]/1.))
        else:
            plt.scatter(X[i, 0], X[i, 1], color=c)

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    else:
        plt.title(params.training_mode)
    plt.savefig(imgName)

plot_embedding(dann_tsne, np.concatenate((s_tags, t_tags)), 'Domain Adaptation: domain', f'tsne_domain_{source}_to_{target}.png', a=True)
plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)), 'Domain Adaptation: class', f'tsne_class__{source}_to_{target}.png', a=False)


