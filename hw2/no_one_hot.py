
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import scipy.misc
import scipy.ndimage
import scipy.misc
from network import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from mean_iou_evaluate import read_masks, mean_iou_score
from viz_mask import viz_data
import numpy as np
import scipy.misc
import random
import os
import time
import sys


cls_color = {
    0:  [0, 255, 255],
    1:  [255, 255, 0],
    2:  [255, 0, 255],
    3:  [0, 255, 0],
    4:  [0, 0, 255],
    5:  [255, 255, 255],
    6:  [0, 0, 0],
}

means     = np.array([103.939, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 512, 512
train_h   = int(h/4)  # 512   
train_w   = int(w/4)  # 512
val_h     = h  # 512
val_w     = w  # 512

train_path ="hw2_data/p2_data/train/"
val_path ="hw2_data/p2_data/validation/"
def get_filelist(file_path):
    _file_list = [file for file in os.listdir(file_path) if file.endswith('.jpg')]
    _file_list.sort()
    return _file_list
class P2_Dataset(Dataset):
    def __init__(self, file_path, phase, n_class=7, crop=False, flip_rate=0.):
        
        self.path        = file_path
        self.masks       = read_masks(file_path)       
        self.file_list   = get_filelist(file_path)
        self.means       = means
        self.n_class     = n_class
        self.flip_rate   = flip_rate
        self.crop        = crop
        if phase == 'train':
            self.crop = True
            self.flip_rate = 0.5
            self.new_h = train_h
            self.new_w = train_w
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name   = self.file_list[idx]
        img        = scipy.misc.imread(os.path.join(self.path, img_name), mode='RGB')
        _img = torch.from_numpy(img.copy())
        label      = self.masks[idx]
        if self.crop:
            h, w, _ = img.shape
            top   = random.randint(0, h - self.new_h)
            left  = random.randint(0, w - self.new_w)
            img   = img[top:top + self.new_h, left:left + self.new_w]
            label = label[top:top + self.new_h, left:left + self.new_w]

        if random.random() < self.flip_rate:
            img   = np.fliplr(img)
            label = np.fliplr(label)


        
        # reduce mean
        img = img[:, :, ::-1]  # switch to BGR
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

        # convert to tensor
        img = torch.from_numpy(img.copy()).float()
        label = torch.from_numpy(label.copy()).long()

        # create one-hot encoding
        h, w = label.size()
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1

        sample = {'X': img, 'Y': target, 'l': label ,'o': _img}

        return sample



n_class    = 7

batch_size = 64
epochs     = 500
lr         = 0.00005
momentum   = 0
w_decay    = 1e-6
step_size  = 50
gamma      = 0.5


'''data processing'''

'''if sys.argv[1] == 'CamVid':
    train_data = CamVidDataset(csv_file=train_file, phase='train')
else:'''
train_data = P2_Dataset(file_path = train_path, phase='train')
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data = P2_Dataset(file_path=val_path, phase='val')
val_loader = DataLoader(val_data, batch_size=1)







'''training'''

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
fcn_model = FCN32s(pretrained_net=vgg_model, n_class=n_class)
vgg_model = vgg_model.cuda()
fcn_model = fcn_model.cuda()

fcn_model = torch.nn.DataParallel(fcn_model)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious

def train():
    best_meanIOU = 0.0
    for epoch in range(epochs):
        

        for _iter, batch in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = Variable(batch['X'].cuda())
            labels = Variable(batch['Y'].cuda())

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
               
        fcn_model.eval()
        with torch.no_grad():
            total_ious = []
            pixel_accs = []
            preds       = []
            targets     = []
            for _iter, batch in enumerate(val_loader):
                
                inputs = Variable(batch['X'].cuda())
                
                output = fcn_model(inputs)
                output = output.data.cpu().numpy()

                N, _, h, w = output.shape

                #print(str(N)+"   "+str(output.shape))
                pred = output.transpose(0, 2, 3, 1)
                #print(pred.shape)
                pred = pred.reshape(-1, n_class)
                #print(pred.shape)
                pred = pred.argmax(axis=1)
                #print(pred.shape)
                pred = pred.reshape(N, h, w)
                #print(pred.shape)
                print(pred)                 
                preds.append(np.array(pred).astype(np.int32))

                truth_label_1_H_W = batch['l'].cpu().numpy().reshape(N, h, w)
                targets.append(np.array(truth_label_1_H_W))
                for p, t in zip(pred, truth_label_1_H_W):                   
                    correct = (pred == truth_label_1_H_W).sum()
                    total   = (truth_label_1_H_W == truth_label_1_H_W).sum()
                    pixel_accs.append(correct/total)
                    total_ious.append(iou(p, t))
                    
                cs = np.unique(np.array(pred))
                print(cs)
                for v_i in [10,97,107]:
                    if _iter == v_i:
                        for c in cs:
                            mask = np.zeros((batch['o'].shape[1], batch['o'].shape[2]))
                            ind = np.where(int(pred) == int(c))
                            mask[ind[0], ind[1]] = 1
                            print(mask.shape)
                            print(batch['o'].shape)
                            img = viz_data(batch['o']., mask, color=cls_color[c])
                        scipy.misc.imsave('./picture/%d_%d.png'%(epoch,v_i), np.uint8(img))   


            _pixel_acc = np.array(pixel_accs).mean()
            total_ious = np.array(total_ious).T  # n_class * val_len
            _meanIOU =  np.nanmean(np.nanmean(total_ious, axis=1))
            if _meanIOU > best_meanIOU:            
                torch.save(fcn_model.state_dict(),'model_best_p2.pkl')
                best_meanIOU = _meanIOU
            print('Epoch:%d -> mean pixel accuracy: %f, mean_iou: %f'%(epoch,_pixel_acc,_meanIOU))
train()               