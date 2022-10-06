import torch

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models
import numpy as np
import scipy.misc
from scipy.misc import imresize
import matplotlib.pyplot as plt
from PIL import Image
from hw2_p2_network import fcn16s, fcn32s, fcn8s
from mean_iou_evaluate import mean_iou_score

import os


X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_valid = np.load("X_val.npy")
y_valid = np.load("y_val.npy")


X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).long()
X_valid = torch.from_numpy(X_valid).float()

valid_path = "./hw2_data/p2_data/validation/"


valid_filelist = sorted(list(set([item.split("_")[0] for item in os.listdir(valid_path)])))

#hyperparameter
num_class = 7
lr = 0.0001
batch_size = 64
epochs = 10

model_name = "fcn16s"
print(model_name)
total_length = len(X_train)

model = fcn16s(num_class).cuda()
model = torch.nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.parameters(),lr=lr, betas=(0.9, 0.999))
criterion = nn.NLLLoss()

 
IOU_list = []
target_list = [10,97,107]
best_meanIOU = 0.0

# training
for epoch in range(epochs):
         
    running_loss = 0.0
    training_msg = ""
    training_msg += "["+str(epoch+1)+"/"+str(epochs)+"]"
    
     
    perm_index = torch.randperm(total_length)
    train_X_sfl = X_train[perm_index]
    train_y_sfl = y_train[perm_index]
    for index in range(0,total_length ,batch_size):
        if index + batch_size > total_length:
            break
        # zero the parameter gradients
        optimizer.zero_grad()
        input_X = train_X_sfl[index:index+batch_size]
        input_y = train_y_sfl[index:index+batch_size]

        # use GPU
        input_X = Variable(input_X.cuda())
        input_y = Variable(input_y.cuda())

        # forward + backward + optimize
        outputs = model(input_X)
        outputs = F.log_softmax(outputs, dim= 1)
        loss = criterion(outputs, input_y)
        loss.backward()
        optimizer.step()
        #print(loss.data)
        running_loss += loss.item()

    training_msg += " Loss:"+str(running_loss/(total_length/batch_size))    
    
    # validation stage
    model.eval()
    with torch.no_grad():
        pred = torch.FloatTensor()
        pred = pred.cuda()
        for i in range(len(X_valid)):
            input_X_valid = Variable(X_valid[i].view(1,3,256,256).cuda())
            output = model(input_X_valid)
            pred = torch.cat((pred,output.data),0)
        
        pred = pred.cpu().numpy()
        pred = np.argmax(pred,1)
        pred_resize = [] 
        for p in pred:
            B = 2                                       # block size - 2  
            im_resize = np.zeros([p.shape[0]*B,p.shape[1]*B])  # output array - 6x6
            for i,l in enumerate(p):                   # lines in a
                for j,aij in enumerate(l):             # a[i,j]
                    im_resize[B*i:B*(i+1),B*j:B*(j+1)] = aij
            pred_resize.append(im_resize)               
        #print(pred_resize[0])
        pred_resize = np.array(pred_resize)
        mean_iou = mean_iou_score(pred_resize, y_valid)
        IOU_list.append(mean_iou)

        training_msg += ", Mean iou score: " + str(mean_iou)
        print(training_msg)
        
        if epoch+1 in [1,10,20]: # save pred map
            n_masks = len(X_valid)
            masks_decoded = np.empty((3, 512, 512, 3))
            i = 0
            for mask, _pred in enumerate(pred_resize):            
                if mask in target_list:
                    masks_decoded[i, _pred == 0] = [  0, 255, 255]
                    masks_decoded[i, _pred == 1] = [255, 255,   0]
                    masks_decoded[i, _pred == 2] = [255,   0, 255]
                    masks_decoded[i, _pred == 3] = [  0, 255,   0]
                    masks_decoded[i, _pred == 4] = [  0,   0, 255]
                    masks_decoded[i, _pred == 5] = [255, 255, 255]
                    masks_decoded[i, _pred == 6] = [  0,   0,   0]
                    i += 1
            masks_decoded = masks_decoded.astype(np.uint8)
            for it, mask_RGB in enumerate(masks_decoded):
                scipy.misc.imsave(os.path.join("./picture/"+model_name+"_"+str(epoch+1)+"_"+valid_filelist[target_list[it]]+"_mask.png"), mask_RGB)
        if mean_iou > best_meanIOU:
            best_meanIOU = mean_iou
            #torch.save(model.state_dict(), model_name+str(mean_iou)[:5]+"_p2_improved_model.pkl")
    model.train()
