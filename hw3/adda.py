import os
import sys
from os import listdir
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from os.path import isfile, join
from PIL import Image
import numpy as np
from torch.autograd import Function
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
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

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)

def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)

def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()
    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists("snapshots"):
        os.makedirs("snapshots")
    torch.save(net.state_dict(),
               os.path.join("snapshots", filename))
    print("save pretrained model to: {}".format(os.path.join("snapshots", filename)))

class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat

class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

def train_src(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-4,
        betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################
    best = 0.0
    num_epochs = 52
    for epoch in range(num_epochs):
        
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % int(len(data_loader)/10) == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}".format(epoch + 1, num_epochs, step + 1, len(data_loader), loss.item()), end = '\r')
        # eval model on test set
        print('')
        if ((epoch + 1) % 4 == 0):
            acc = eval_src(encoder, classifier, data_loader)
            # # save final model
            if acc > best:
                save_model(encoder, f"ADDA-source-encoder-final_{source}_to_{target}.pt")
                save_model(classifier, f"ADDA-source-classifier-final_{source}_to_{target}.pt")
                best = acc

    return encoder, classifier
def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images)
        labels = make_variable(labels)

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        pred_cls = preds.max(1)[1]
        acc += pred_cls.eq(labels).cpu().sum()
        

    loss /= len(data_loader)
    acc = acc.float()
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))
    return acc
def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, tgt_data_loader_eval,classifier):
    """Train encoder for target domain."""
    # 1. setup network #
    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()
    classifier.eval()
    # setup criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))   
    # 2. train network #
    num_epochs_tg = 150
    best = 0.0
    for epoch in range(num_epochs_tg):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))        
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            # 2.1 train discriminator #
            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)
            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)
            # predict on discriminator
            pred_concat = critic(feat_concat.detach())
            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)
            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()
            # optimize critic
            optimizer_critic.step()
            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()
            # 2.2 train target encoder #
            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()
            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)
            # predict on discriminator
            pred_tgt = critic(feat_tgt)
            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())
            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()
            # optimize target encoder
            optimizer_tgt.step()
            # 2.3 print step info #
            if ((step + 1) % 2 == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1, num_epochs_tg, step + 1, len_data_loader, loss_critic.item(), loss_tgt.item(), acc.item()), end='\r')
        print('')
        # 2.4 save model parameters #
        if ((epoch + 1) % 1 == 0):
            acc, _ = eval_tgt(tgt_encoder, classifier, tgt_data_loader_eval)
            # # save final model
            if acc > best:
                print("saving...")
                save_model(critic,f"ADDA-critic-final_{source}_to_{target}_last.pt")
                save_model(tgt_encoder,f"ADDA-target-encoder-final_{source}_to_{target}_last.pt")
                best = acc     
    
    return tgt_encoder
def eval_tgt(encoder, classifier, data_loader, prediction = False):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0.0
    acc = 0.0

    # set loss function
    criterion = nn.NLLLoss()

    # evaluate network
    result = []
    for images in data_loader:
        images = make_variable(images)

        preds = classifier(encoder(images))
        if prediction:
            x = torch.argmax(preds, dim=1).cpu().detach().numpy()
            result.append(x)

    return result

class dataReader(Dataset):
    def __init__(self, root, transforms, train = True):
        self.root = root
        self.train = train        
        self.transforms = transforms
        self.onlyfiles = [f for f in listdir(self.root) if isfile(join(self.root, f))]
        self.length = len(self.onlyfiles)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.onlyfiles[idx]))
        img = self.transforms(img)
        return img
    
    def __len__(self):
        return self.length

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

root = sys.argv[1]#"./hw3_data/digits/usps/test"
target = sys.argv[2]#"usps"
target_path = os.path.join(root)
source = ""
if target == "mnistm":
    source = "usps"
elif target == "usps":
    source = "svhn"
else:
    source = "mnistm"

test_target_dataset = dataReader(target_path, transforms=test_transform, train = False)
tgt_data_loader_eval = DataLoader(test_target_dataset, batch_size = 128, shuffle=False)

dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 64
image_size = 28

# params for setting up models
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2

# params for training network
num_gpu = 1
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
log_step = 100
save_step = 100
manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

# load models
src_encoder = init_model(net=LeNetEncoder(),
                            restore=None)
src_classifier = init_model(net=LeNetClassifier(),
                            restore=None)
tgt_encoder = init_model(net=LeNetEncoder(),
                            restore=None)
critic = init_model(Discriminator(input_dims=d_input_dims,
                                    hidden_dims=d_hidden_dims,
                                    output_dims=d_output_dims),
                    restore=None)

# train source model
print("=== Training classifier for source domain ===")
#print(">>> Source Encoder <<<")
#print(src_encoder)
#print(">>> Source Classifier <<<")
#print(src_classifier)
'''
if not (src_encoder.restored and src_classifier.restored):
    src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader)
 
src_encoder.load_state_dict(torch.load(os.path.join(
                   "snapshots",
                    f'ADDA-source-encoder-final_{source}_to_{target}.pt')))

# eval source model
print("=== Evaluating classifier for source domain ===")
_ = eval_src(src_encoder, src_classifier, src_data_loader_eval)

# train target encoder by GAN
print("=== Training encoder for target domain ===")
#print(">>> Target Encoder <<<")
#print(tgt_encoder)
#print(">>> Critic <<<")
#print(critic)

# init weights of target encoder with those of source encoder
tgt_encoder.load_state_dict(torch.load(os.path.join(
                   "snapshots",
                    f'ADDA-source-encoder-final_{source}_to_{target}.pt')))
tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader, tgt_data_loader_eval, src_classifier)
'''
src_classifier.load_state_dict(torch.load(os.path.join(
                    "snapshots",
                    f'ADDA-source-classifier-final_{source}_to_{target}.pt?dl=1')))                               
tgt_encoder.load_state_dict(torch.load(os.path.join(
                    "snapshots",
                    f'ADDA-target-encoder-final_{source}_to_{target}_last.pt?dl=1')))
result = eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval, prediction=True)
import pandas as pd
result = np.concatenate(result)
onlyfiles = [f for f in listdir(root) if isfile(join(root, f))]
# Generate your submission
df = pd.DataFrame({'image_name': onlyfiles, 'label': result})
df.to_csv(sys.argv[3],index=False)

'''

from sklearn.manifold import TSNE
num_of_samples = 2048
s_images, s_labels, s_tags = [], [], []
t_images, t_labels, t_tags = [], [], []
for i, ((source_data, source_label), (target_data, target_label)) in enumerate(zip(src_data_loader_eval, tgt_data_loader_eval)):
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
embedding1 = src_encoder(s_images)
embedding2 = tgt_encoder(t_images)

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