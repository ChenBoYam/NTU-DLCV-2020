import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import torch
import sys
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import random
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchsummary import summary


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#workspace_dir = os.path.join(sys.argv[1])
#output_dir = os.path.join(sys.argv[2])

same_seeds(0)
class dataReader(Dataset):
    def __init__(self, root, transforms):
        self.root = root 
        self.transforms = transforms
        self.num = []
        onlyfiles = [f for f in listdir(self.root) if isfile(join(self.root, f))]
        self.length = len(onlyfiles)
        for _str in onlyfiles:
            self.num.append(_str.split(".",1)[0])

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, str(self.num[idx] + '.png')))
        img = self.transforms(img)
        return img
    
    def __len__(self):
        return self.length

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([                                   
    transforms.ToTensor(),
])



batch_size = 32

train_set = dataReader('hw3_data/face/train', train_transform)
test_set = dataReader('hw3_data/face/test', test_transform)               
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_set, batch_size = 10, shuffle=False)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=100):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=12544, z_dim=100):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(z_dim, 512, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 64, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, log_var = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def encode(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        _z = self.decode(z)
        return _z, mu, log_var ,z
def loss_function(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    MSE = criterion(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return MSE + KLD, MSE ,KLD

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64 )
    return x

def plot_with_labels(lowDWeights, labels):
    plt.cla() #clear当前活动的坐标轴
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1] #把Tensor的第1列和第2列,也就是TSNE之后的前两个特征提取出来,作为X,Y
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 50))
        #plt.text(x, y, s, backgroundcolor=c, fontsize=9)
        plt.scatter(x, y, color = c, label = s, s = 1) #在指定位置放置文本
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize second last layer')
    plt.savefig("tsne_new.png")


model = VAE()
if torch.cuda.is_available():
    model.cuda()
print(model)
'''
criterion = nn.MSELoss(size_average=False)

optimizer = optim.Adam(model.parameters(), lr=2e-4)

mse_list = []
kld_list = []
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    mse_loss = 0
    kld_loss = 0
    for batch_idx, img in enumerate(train_loader):
        img = Variable(img)
        if torch.cuda.is_available():
            img = img.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(img)
        loss, mse, kld = loss_function(recon_batch, img, mu, logvar, criterion)        
        loss.backward()
        train_loss += loss.item()
        mse_loss += mse
        kld_loss += kld
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(img),
                len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item() / len(img)))
    mse_list.append(mse_loss)
    kld_list.append(kld_loss)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))
    if epoch % 1 == 0:
        save = to_img(recon_batch.cpu().data)
        save_image(save, './vae_img/image_{}.png'.format(epoch))
    if (epoch+1) % 10 == 0:   
        torch.save(model.state_dict(), './vae_{}.pkl'.format(epoch))
plt.plot(range(1,num_epochs+1,1), mse_list)
plt.title('MSE loss')
plt.xlabel('epochs')
plt.ylabel('mse')
plt.savefig('MSE_loss.png')
plt.clf()
plt.plot(range(1,num_epochs+1,1), kld_list)
plt.title('KLD loss')
plt.xlabel('epochs')
plt.ylabel('kld')
plt.savefig('KLD_loss.png') 
'''

model.load_state_dict(torch.load('./model/vae_40.pkl'))
model.eval()
for batch_idx, img in enumerate(test_loader):
        if batch_idx == 0:
            img = Variable(img)
            if torch.cuda.is_available():
                img = img.cuda()
            recon_batch, mu, logvar, _ = model(img)
            loss, mse, kld = loss_function(recon_batch, img, mu, logvar, criterion)        
            mse_list.append(mse)
            save = to_img(recon_batch.cpu().data)
            save_image(save, './picture/image_test.png')

model.eval()
img = torch.randn(32, 3, 64, 64).cuda()
if torch.cuda.is_available():
    img = img.cuda()
recon_batch, mu, logvar = model(img)
loss, mse, kld = loss_function(recon_batch, img, mu, logvar, criterion)        
mse_list.append(mse)
save = to_img(recon_batch.cpu().data)
save_image(save, './picture/fig1_4.jpg')

'''
from sklearn.manifold import TSNE
num_of_samples = 2048
s_images, s_labels, s_tags = [], [], []
t_images, t_labels, t_tags = [], [], []
for i, ((source_data, source_label), (target_data, target_label)) in enumerate(zip(train_loader, test_loader)):
    s_images.append(source_data.cuda())
    s_labels.append(source_label)
    s_tags.append(torch.zeros((source_label.size()[0])).type(torch.LongTensor))
    t_images.append(target_data.cuda())
    t_labels.append(target_label)
    t_tags.append(torch.ones((target_label.size()[0])).type(torch.LongTensor))

s_images, s_tags, s_labels = torch.cat(s_images)[:num_of_samples], torch.cat(s_tags)[:num_of_samples],torch.cat(s_labels)[:num_of_samples]
t_images, t_tags, t_labels = torch.cat(t_images)[:num_of_samples], torch.cat(t_tags)[:num_of_samples], torch.cat(t_labels)[:num_of_samples]
s_images = make_variable(s_images)
t_images = make_variable(t_images)
_, _, _, embedding1 = model(s_images)
_, _, _, embedding2 = model(t_images)

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

plot_embedding(dann_tsne, np.concatenate((s_tags, t_tags)), 'Domain Adaptation: domain', f'ADDA_tsne_domain_{source}_to_{target}.png', a=True)
plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)), 'Domain Adaptation: class', f'ADDA_tsne_class__{source}_to_{target}.png', a=False)

'''
