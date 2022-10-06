import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet,Hallusnator,Discriminator
from utils import pprint, ensure_path, Averager, Timer, count_acc, euclidean_metric
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=1)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--hul-num', type=int, default=200)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto_1')
    # parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    # set_gpu(args.gpu)
    ensure_path(args.save_path)

    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers = 3, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers = 3, pin_memory=True)

    model = Convnet().cuda()
    model_h = Hallusnator().cuda()
    model_d = Discriminator().cuda()

    print(model)
    print(model_h)
    print(model_d)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    optimizer_d = torch.optim.Adam(model_d.parameters(), lr=0.001)
    lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=20, gamma=0.5)


    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    loss_all = []
    acc_all = []
    acc_all_val=[]
    acc_train_max = 0
    for epoch in range(1, args.max_epoch + 1):
        

        model.train()
        model_d.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
         
            idx_of_feature = np.random.randint(args.train_way)
            shot = proto[idx_of_feature]
            
            proto_all = []
            for x in range(args.hul_num):
                
                guass = torch.randn(512).cuda()
                hul = torch.cat((shot,guass))

                proto_h = model_h(hul)
                proto_all.append(proto_h)
            
            proto_temp = torch.zeros(1600)
            proto_temp = proto_temp.cuda()

            for xx in range(args.hul_num) : 
                proto_temp = proto_temp + proto_all[xx]
            
            proto_fake = (proto_temp + proto[idx_of_feature]) / (args.hul_num + 1 ) 
            proto_true = proto[idx_of_feature]

            out_fake = model_d(proto_fake)
            out_fake = out_fake.unsqueeze(0)
            out_true = model_d(proto_true) 
            out_true = out_true.unsqueeze(0)
            out = torch.cat((out_true,out_fake),0)

            real_label = torch.ones(1).cuda()
            fake_label = torch.zeros(1).cuda()
            label = torch.cat((real_label,fake_label),0)

            loss_d = nn.BCELoss()(out,label)

            optimizer_d.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizer_d.step()

            loss_h = nn.BCELoss()(out_fake,real_label)
            loss_h.backward(retain_graph=True)
            optimizer_h.step()

            proto[idx_of_feature] = (proto_temp + proto[idx_of_feature]) / (args.hul_num + 1 ) 

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss_d={:.4f} loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader),loss_d, loss.item(), acc))

            loss_all.append(loss.item())
            acc_all.append(acc)
            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None
            if i % 10 == 0:
                save_model('epoch-{}-{}'.format(epoch,i))
                if(acc > acc_train_max) : 
                    acc_train_max = acc
                    print(epoch,i)
                    eepoch = epoch
                    ii = i


        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)

            print(i, end='\r')
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        acc_all_val.append(va)

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
        
        
        plt.figure(figsize=(10,5))
        plt.title("lost_train%d_hul%d"%(args.max_epoch,args.hul_num))
        plt.plot(loss_all)
        plt.savefig(osp.join(args.save_path,"lost_train%d_hul%d.jpg"%(args.max_epoch,args.hul_num)))
        # plt.savefig("./save/proto-2_big_guassion/lost_train%d_hul%d.jpg"%(args.max_epoch,args.hul_num))

        plt.figure(figsize=(10,5))
        plt.title("accuracy_train%d_hul%d"%(args.max_epoch,args.hul_num))
        plt.plot(acc_all)
        plt.savefig(osp.join(args.save_path,"accuracy_train%d_hul%d.jpg"%(args.max_epoch,args.hul_num)))
        # plt.savefig("./save/proto-2_big_guassion/accuracy_train%d_hul%d.jpg"%(args.max_epoch,args.hul_num))
        

        plt.figure(figsize=(10,5))
        plt.title("accuracy_train%d_hul%d"%(args.max_epoch,args.hul_num))
        plt.plot(acc_all_val)
        plt.savefig(osp.join(args.save_path,"accuracy_val_train%d_hul%d.jpg"%(args.max_epoch,args.hul_num)))
        # plt.savefig("./save/proto-2_big_guassion/accuracy_val_train%d_hul%d.jpg"%(args.max_epoch,args.hul_num))
