import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from mini_imagenet import MiniImageNet
from data_sample import CategoriesSampler
from network import Convnet,Hallusnator
from utils import pprint, ensure_path, Averager, Timer, count_acc, euclidean_metric
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from PIL import Image
from torchvision import transforms

if __name__ == '__main__':
    


    id_class = []
    while(len(id_class) < 5):
        temp = np.random.randint(64)
        if(temp not in id_class) : id_class.append(temp)

    
    class_id_200 = []
    while(len(class_id_200) < 200):
        temp = np.random.randint(600)
        if(temp not in id_class) : class_id_200.append(temp)
    
    train_csv = pd.read_csv("./hw4_data/train.csv").set_index("id")
    train_csv = train_csv.values.tolist()

    tsne_csv = []
    for i in range(5) : 
        for ii in range(200) : 
            tsne_csv.append(train_csv[id_class[i]*600+class_id_200[ii]])

    all_image = torch.zeros(1000,3,84,84)

    image_transform = transforms.Compose([
                transforms.Resize(84),
                transforms.CenterCrop(84),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

    img_path = './hw4_data/train'
    for i in range(1000) : 
        path = osp.join(img_path,tsne_csv[i][0])
        image = image_transform(Image.open(path).convert('RGB'))
        all_image[i,:,:,:] = image
    
    all_image = all_image.cuda()


    model = Convnet().cuda()
    model.load_state_dict(torch.load("./save/proto-h-d-4/max-acc.pkl"))
    model.eval()

    model_h = Hallusnator().cuda()

    # print(model)
    # print(model_h)
    
    all_image_cpu = []
    for i in range(1000):
        temp = torch.zeros(1,3,84,84).cuda()
        temp[0] = all_image[i]
        y = model(temp)
        y = y.cpu().detach().numpy()
        all_image_cpu.append(y)


    
    h_all = torch.zeros(100,1600).cuda()
    
    guass = torch.randn((100,200)).cuda()
    # guass = torch.randn((100,2112)).cuda()
    # guass = torch.randn((20,512)).cuda()
    idx = []
    while(len(idx) < 20) : 
        temp = np.random.randint(200)
        if temp not in idx:
            idx.append(temp)
    
    for i in range(5) :
        for ii in range(20) :
            k = all_image_cpu[i*200 + idx[ii]]
            k = torch.from_numpy(k) 
            k = k.cuda() 
            k = k.squeeze(dim=0)
            # hul = torch.cat((k,guass[i%2]))
            # hul = torch.cat((k,guass[ii]))
            hul = torch.cat((k,guass[i*20+ii]))
            # hul = guass[ii]
            # hul = guass[i*20+ii]
            proto_h = model_h(hul)
            h_all[i*20+ii] = proto_h       

    h_all = h_all.cpu().detach().numpy()
    
    image_cpu = np.zeros((1000,1600))
    for i in range(1000) : 
        image_cpu[i] = all_image_cpu[i]
    
    z = np.concatenate((image_cpu,h_all))
            
    z_embedded = TSNE(n_components=2).fit_transform(z)

    
    for i in range(1000):
        if(i<200):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='x',c="red")
        elif(200<=i and i<400):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='x',c="green")
        elif(400<=i and i<600):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='x',c="yellow")
        elif(600<=i and i<800):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='x',c="blue")
        elif(800<=i and i<1000):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='x',c="purple")
        
    for i in range(1000,1100) :        
        if(1000<=i and i<1020):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='^',c="red")
        elif(1020<=i and i<1040):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='^',c="green")
        elif(1040<=i and i<1060):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='^',c="yellow")
        elif(1060<=i and i<1080):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='^',c="blue")
        elif(1080<=i and i<1100):
            plt.scatter(z_embedded[i, 0], z_embedded[i, 1],marker='^',c="purple")
        
    plt.plot()
    plt.savefig('tsne-d-2.jpg')
    plt.show()
        

