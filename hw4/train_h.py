import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data_sample import CategoriesSampler
from mini_imagenet import MiniImageNet
from network import Convnet, Hallusnator
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--train-way', type=int, default=30)
    parser.add_argument('--train-hul', type=int, default=200)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save/proto-h-2')
    args = parser.parse_args()
    pprint(vars(args))

    ensure_path(args.save_path)
    '''./hw4_data/train.csv','hw4-data/train/'''
    trainset = MiniImageNet('train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=3, pin_memory=True)

    valset = MiniImageNet('val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=3, pin_memory=True)

    model = Convnet().cuda()
    model_hal = Hallusnator().cuda()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pkl'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            #print(proto.shape)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            hal_i = np.random.randint(args.train_way)
            proto_h = torch.zeros(1600).cuda()
            for m in range(args.train_hul):
                
                noise = torch.randn(400).cuda()
                hul_input = torch.cat((proto[hal_i],noise))

                proto_h += model_hal(hul_input)#.unsqueeze(0))
                #print(proto_h.shape)
            
            loss_h = nn.BCELoss()(out_fake,real_label)
            loss_h.backward(retain_graph=True)
            optimizer_h.step()

            proto[hal_i] = (proto_h + proto[hal_i])/(args.train_hul+1)

            

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)           
            q = model(data_query)
            #print(f"proto:{proto.shape}, label:{label.shape}, model(data_query):{q.shape}")
            logits = euclidean_metric(q, proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print(f'epoch {epoch}, train {i}/{len(train_loader)}, loss={loss.item()} acc={acc}', end='\r')

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None

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
            
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

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
    plt.title("Tr_Loss_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul))
    plt.plot(trlog['train_loss'])
    plt.savefig(osp.join(args.save_path,"Tr_Loss_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul)))
    # plt.savefig("./save/proto-2_big_guassion/lost_train%d_hul%d.jpg"%(args.max_epoch,args.hul_num))

    plt.figure(figsize=(10,5))
    plt.title("Tr_Acc_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul))
    plt.plot(trlog['train_acc'])
    plt.savefig(osp.join(args.save_path,"Tr_Acc_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul)))
    

    plt.figure(figsize=(10,5))
    plt.title("val_Acc_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul))
    plt.plot(trlog['val_acc'])
    plt.savefig(osp.join(args.save_path,"val_Acc_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul)))
    