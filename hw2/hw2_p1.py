import os
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image
import torch
import time
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg16
from matplotlib import cm
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE


workspace_dir = os.path.join(sys.argv[1])
output_dir = os.path.join(sys.argv[2])


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
        img = Image.open(os.path.join(self.root, self.num[idx] + '.png'))
        img = self.transforms(img)
        return img
    
    def __len__(self):
        return self.length


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([                                   
    transforms.ToTensor(),
])
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

batch_size = 32

#train_set = dataReader('hw2_data/p1_data/train_50', train_transform)
val_set = dataReader(workspace_dir, test_transform)               
#train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
val_loader = DataLoader(val_set, batch_size = batch_size, shuffle=False)

'''
model = vgg16(pretrained = True).cuda()

loss = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001) 
num_epoch = 80
best_acc = 0.0
for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() 
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda()) 
        batch_loss.backward()
        optimizer.step() 

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()
        if val_acc > best_acc:            
            torch.save(model.state_dict(),'model_best.pkl')
            best_acc = val_acc
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))
'''


#from torchsummary import summary


model = vgg16().cuda()
#summary(model, (3, 32, 32))
model.load_state_dict(torch.load('model_best.pkl?dl=1'))

"""# Testing
利用剛剛 train 好的 model 進行 prediction
"""
'''
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('matrix.png')
'''
model.eval()
ground_truth = []
prediction = []
val_acc = 0.0
with torch.no_grad():
    for i, data in enumerate(val_loader):
        val_pred = model(data.cuda())
        #last_second = model.features[:30](data[0].cuda())
        #last_second = last_second.view(last_second.size(0),-1).cpu().data.numpy()
        #if i == 0:
        #    last_seconds = last_second
        #    labels = data[1].numpy()
        #else:
        #    last_seconds = np.concatenate((last_seconds,last_second),axis = 0)
        #    labels =  np.concatenate((labels,data[1].numpy()),axis = 0)
        val_label = np.argmax(val_pred.cpu().data.numpy(), axis=1)
        #val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        
            
        
        for y in val_label:
            prediction.append(y)

    #t-SNE 是一种非线性降维算法，非常适用于高维数据降维到2维或者3维，进行可视化
    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)


    #low_dim_embs = tsne.fit_transform(last_seconds[:1500, :])
    #plot出來
#   plot_with_labels(low_dim_embs, labels)
    


    #print('Val Acc: %3.6f' % (val_acc/val_set.__len__()))



''' 
cm = confusion_matrix(val_set.get_label(), prediction)
names = [str(f) for f in range(1,51)]
plt.figure(figsize = (50,50))
plot_confusion_matrix(cm, names,True)
'''

filename = [f for f in listdir(workspace_dir) if isfile(join(workspace_dir, f))]
zip_iterator = zip(filename, prediction)
_dict = dict(zip_iterator)

with open(output_dir+"test_pred.csv", 'w') as f:
    f.write('image_id,label\n')
    for png in  _dict:
        f.write('{},{}\n'.format(png, _dict[png]))
