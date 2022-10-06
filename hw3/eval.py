import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import models
import utils
import os
import argparse
from torch.utils.data import DataLoader, Dataset
import csv
import glob
from PIL import Image
import matplotlib.pyplot as plt

class ImgDataset(Dataset):
    def __init__(self, root,root_label, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.labels = []
        self.filenames = []
        self.root = root
        self.transform = transform
    
        # self.filenames = glob.glob(os.path.join(root,'*.png'))
        
        with open(root_label, newline='') as csvfile:
            rows = csv.reader(csvfile)
            
            for row in rows:
                self.labels.append(row[1])
                self.filenames.append(root+row[0])
                # print(row)
            del(self.labels[0])
            del(self.filenames[0])
            

        self.len = len(self.filenames)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image = Image.open(self.filenames[index])
        label = self.labels[index]
        
        if self.transform is not None:
            image = self.transform(image)
            label = int(label)
        
        return image,label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


mean = np.array([0.44, 0.44, 0.44])
std = np.array([0.19, 0.19, 0.19])

# img_transform_source = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean,std)])
# img_transform_target = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean,std)])

img_transform_source = transforms.Compose([transforms.Grayscale(3),transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean,std)])
img_transform_target = transforms.Compose([transforms.Grayscale(3),transforms.Resize(32), transforms.ToTensor(), transforms.Normalize(mean,std)])

batch_size = 100


root ="./digits/usps/test/"
root_label = './digits/usps/test.csv'

dataset_source = ImgDataset(root,root_label,transform=img_transform_target)
print('# images in  dataset_target:', len(dataset_source)) 

dataloader_source= DataLoader(dataset_source, batch_size=batch_size, shuffle=True, num_workers=0)
print(len(dataloader_source),"dataloader_target length")  

root ="./digits/mnistm/test/"
root_label = './digits/mnistm/test.csv'

dataset_target = ImgDataset(root,root_label,transform=img_transform_target)
print('# images in  dataset_target:', len(dataset_target)) 

dataloader_target= DataLoader(dataset_target, batch_size=batch_size, shuffle=True, num_workers=0)
print(len(dataloader_target),"dataloader_target length")   




def main():

    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataroot', required=True, help='path to source dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Number of filters to use in the generator network')
    parser.add_argument('--ndf', type=int, default=64, help='Number of filters to use in the discriminator network')
    parser.add_argument('--gpu', type=int, default=1, help='GPU to use, -1 for CPU training')
    parser.add_argument('--checkpoint_dir', default='results/models_s2u', help='folder to load model checkpoints from')
    parser.add_argument('--method', default='GTA', help='Method to evaluate| GTA, sourceonly')
    parser.add_argument('--model_best', type=int, default=0, help='Flag to specify whether to use the best validation model or last checkpoint| 1-model best, 0-current checkpoint')

    opt = parser.parse_args()

    # GPU/CPU flags
    cudnn.benchmark = True
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --gpu [gpu id]")
    if opt.gpu>=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    # Creating data loaders
    # mean = np.array([0.44, 0.44, 0.44])
    # std = np.array([0.19, 0.19, 0.19])

    # target_root = os.path.join(opt.dataroot, 'mnist/trainset')

    # transform_target = transforms.Compose([transforms.Resize(opt.imageSize), transforms.ToTensor(), transforms.Normalize(mean,std)])
    # target_test = dset.ImageFolder(root=target_root, transform=transform_target)
    # targetloader = torch.utils.data.DataLoader(target_test, batch_size=opt.batchSize, shuffle=False, num_workers=2)

    targetloader = dataloader_target
    # nclasses = len(target_test.classes)
    nclasses = 10
    
    # Creating and loading models
    
    netF = models._netF(opt)
    netC = models._netC(opt, nclasses)
    
    if opt.method == 'GTA':
        netF_path = os.path.join(opt.checkpoint_dir, 'netF_10_140.pth')
        netC_path = os.path.join(opt.checkpoint_dir, 'netC_10_140.pth')
        # if opt.model_best == 0: 
        #     netF_path = os.path.join(opt.checkpoint_dir, 'netF.pth')
        #     netC_path = os.path.join(opt.checkpoint_dir, 'netC.pth')
        # else:
        #     netF_path = os.path.join(opt.checkpoint_dir, 'model_best_netF.pth')
        #     netC_path = os.path.join(opt.checkpoint_dir, 'model_best_netC.pth')
    
    elif opt.method == 'sourceonly':
        if opt.model_best == 0: 
            netF_path = os.path.join(opt.checkpoint_dir, 'netF_sourceonly.pth')
            netC_path = os.path.join(opt.checkpoint_dir, 'netC_sourceonly.pth')
        else:
            netF_path = os.path.join(opt.checkpoint_dir, 'model_best_netF_sourceonly.pth')
            netC_path = os.path.join(opt.checkpoint_dir, 'model_best_netC_sourceonly.pth')
    else:
        raise ValueError('method argument should be sourceonly or GTA')
        
    netF.load_state_dict(torch.load(netF_path))
    netC.load_state_dict(torch.load(netC_path))
    
    if opt.gpu>=0:
        netF.cuda()
        netC.cuda()
        
    # Testing
    
    netF.eval()
    netC.eval()
        
    total = 0
    correct = 0
    # print(len(targetloader))
    for i, datas in enumerate(targetloader):
        inputs, labels = datas
        if opt.gpu>=0:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputv, labelv = Variable(inputs, volatile=True), Variable(labels)

        outC = netC(netF(inputv))
        _, predicted = torch.max(outC.data, 1)        
        total += labels.size(0)
        correct += ((predicted == labels.cuda()).sum())
        
    test_acc = 100*float(correct)/total
    print('Test Accuracy: %f %%' % (test_acc))

from sklearn.manifold import TSNE
from torch.autograd import Variable
def tsne():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataroot', required=True, help='path to source dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Number of filters to use in the generator network')
    parser.add_argument('--ndf', type=int, default=64, help='Number of filters to use in the discriminator network')
    parser.add_argument('--gpu', type=int, default=1, help='GPU to use, -1 for CPU training')
    parser.add_argument('--checkpoint_dir', default='results/models_u2m', help='folder to load model checkpoints from')
    parser.add_argument('--method', default='GTA', help='Method to evaluate| GTA, sourceonly')
    parser.add_argument('--model_best', type=int, default=0, help='Flag to specify whether to use the best validation model or last checkpoint| 1-model best, 0-current checkpoint')

    opt = parser.parse_args()

    # GPU/CPU flags
    cudnn.benchmark = True
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --gpu [gpu id]")
    if opt.gpu>=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    dataloader_target_TSNE = dataloader_target
    dataloader_source_TSNE = dataloader_source
    nclasses = 10
    
    # Creating and loading models
    
    netF = models._netF(opt)
    netC = models._netC(opt, nclasses)
    
    if opt.method == 'GTA':
        netF_path = os.path.join(opt.checkpoint_dir, 'dann_model_best_netF.pth')
        netC_path = os.path.join(opt.checkpoint_dir, 'netC_8_240.pth')
            
    netF.load_state_dict(torch.load(netF_path))
    netC.load_state_dict(torch.load(netC_path))
    
    if opt.gpu>=0:
        netF.cuda()
        netC.cuda()
        
    # Testing
    
    netF.eval()
    netC.eval()
    
    dataiter_source = iter(dataloader_source_TSNE)
    dataiter_target = iter(dataloader_target_TSNE)

    images_s, labels_s = dataiter_source.next()
    images_t, labels_t = dataiter_target.next()

    images_s = Variable(images_s)
    images_t = Variable(images_t)

    images_s = images_s.cuda()
    images_t = images_t.cuda()

    input_data = images_s
    feature = netF(input_data)
    z = feature
    labels_all = labels_s

    input_data = images_t
    feature = netF(input_data)
    zz = feature
    z = torch.cat((z,zz),0)

    labels_all = torch.cat((labels_all,labels_t),0)

    aa_domanin = np.zeros(batch_size)
    a = np.zeros(batch_size)
    b = np.zeros(batch_size)
    for t in range(batch_size):
        b[t] = 1

    aa_domanin = np.concatenate((aa_domanin,b),0)

    i=1

    # while i < len(dataloader_source_TSNE):
    while i < 20:

        images_s, labels_s = dataiter_source.next()                                                                                                                                                                                                                                 
        images_t, labels_t = dataiter_target.next()
        
        images_s = Variable(images_s)
        images_t = Variable(images_t)
        
        images_s = images_s.cuda()
        images_t = images_t.cuda()

        input_data = images_s
        feature = netF(input_data)
        zz = feature
        z = torch.cat((z,zz),0)
        labels_all = torch.cat((labels_all,labels_s),0)
        aa_domanin = np.concatenate((aa_domanin,a),0)

        input_data = images_t
        feature = netF(input_data)
        zz = feature
        z = torch.cat((z,zz),0)
        labels_all = torch.cat((labels_all,labels_t),0)
        aa_domanin = np.concatenate((aa_domanin,b),0)


        i += 1

    zzz = TSNE(n_components=2).fit_transform(z.detach().cpu().clone().numpy())

    z_min, z_max = zzz.min(0), zzz.max(0)
    z_norm = (zzz - z_min) / (z_max - z_min)  # 归一化
    # z_norm = zz

    for i in range(len(labels_all)):
        if(labels_all[i].numpy() == 0):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='black')
        elif(labels_all[i].numpy() == 1):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='red')
        elif(labels_all[i].numpy() == 2):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='green')
        elif(labels_all[i].numpy() == 3):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='pink')
        elif(labels_all[i].numpy() == 4):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='gray')
        elif(labels_all[i].numpy() == 5):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='purple')
        elif(labels_all[i].numpy() == 6):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='maroon')
        elif(labels_all[i].numpy() == 7):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='yellow')
        elif(labels_all[i].numpy() == 8):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='brown')
        elif(labels_all[i].numpy() == 9):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='blue')



    plt.show()

    for i in range(len(labels_all)):
        if(aa_domanin[i] == 0):
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='blue')
        else:
            plt.scatter(z_norm[i, 0], z_norm[i, 1], c='red')


    plt.show()

        



if __name__ == '__main__':
    # main()
    tsne()

