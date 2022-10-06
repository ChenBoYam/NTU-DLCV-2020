import torch
import sys
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
from hw2_p2_network import fcn16s, fcn32s
from mean_iou_evaluate import mean_iou_score

import os


valid_path = sys.argv[1]
output_dir = sys.argv[2]

file_list = sorted(list(set([file.split("_")[0] for file in os.listdir(valid_path)]))) 
imgs = []
for i, file in enumerate(file_list):
    img    = scipy.misc.imread(os.path.join(valid_path, file + "_sat.jpg"))
    imgs.append(img)

X_valid = ((np.array(imgs)[:,::2,::2,:])/255).transpose(0,3,1,2)
X_valid = torch.from_numpy(X_valid).float()


model_name = "fcn32s"
print(model_name)

num_class = 7
model = fcn32s(num_class).cuda()
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('fcn32s_p2_model.pkl?dl=1'))


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
    
    masks_decoded = np.empty((len(X_valid), 512, 512, 3))
    for mask, _pred in enumerate(pred_resize):            
        masks_decoded[mask, _pred == 0] = [  0, 255, 255]
        masks_decoded[mask, _pred == 1] = [255, 255,   0]
        masks_decoded[mask, _pred == 2] = [255,   0, 255]
        masks_decoded[mask, _pred == 3] = [  0, 255,   0]
        masks_decoded[mask, _pred == 4] = [  0,   0, 255]
        masks_decoded[mask, _pred == 5] = [255, 255, 255]
        masks_decoded[mask, _pred == 6] = [  0,   0,   0]
    masks_decoded = masks_decoded.astype(np.uint8)
    for it, mask_RGB in enumerate(masks_decoded):
        scipy.misc.imsave(os.path.join(output_dir+file_list[it]+"_mask.png"), mask_RGB)