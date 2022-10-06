import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy
import scipy.misc


def read_data(filepath, train = False):
    '''
    Read masks from directory and tranform to categorical
    '''
    file_list = sorted(list(set([file.split("_")[0] for file in os.listdir(filepath)]))) 
    masks = np.empty((len(file_list), 512, 512))

    imgs = []

    for i, file in enumerate(file_list):
        img    = scipy.misc.imread(os.path.join(filepath, file + "_sat.jpg"))
        imgs.append(img)
        mask = scipy.misc.imread(os.path.join(filepath, file + "_mask.png"))
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]        
        masks[i, mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[i, mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[i, mask == 2] = 3  # (Green: 010) Forest land 
        masks[i, mask == 1] = 4  # (Blue: 001) Water 
        masks[i, mask == 7] = 5  # (White: 111) Barren land 
        masks[i, mask == 0] = 6  # (Black: 000) Unknown
        masks[i, mask == 4] = 6  # (Red: 100) Unknown

    if train:
        masks = masks[:,::2,::2] #downsample in train

    imgs = ((np.array(imgs)[:,::2,::2,:])/255).transpose(0,3,1,2)
    return imgs, masks



train_path = "./hw2_data/p2_data/train/"
valid_path = "./hw2_data/p2_data/validation/"
# construct id list

X_train, y_train = read_data(train_path, train = True)
X_val  , y_val   = read_data(valid_path)

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)
