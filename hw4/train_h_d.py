import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from data_sample import CategoriesSampler
from mini_imagenet import MiniImageNet
from network import Convnet, Hallusnator, Discriminator
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
    parser.add_argument('--save-path', default='./save/proto-h-d-4')
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
    model_disc = Discriminator().cuda()
    print(model)
    print(model_hal)
    print(model_disc)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer_disc = torch.optim.Adam(model_disc.parameters(), lr=0.001)
    optimizer_h = torch.optim.Adam(model_disc.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)   
    lr_scheduler_disc = torch.optim.lr_scheduler.StepLR(optimizer_disc, step_size=20, gamma=0.8)
    lr_scheduler_h = torch.optim.lr_scheduler.StepLR(optimizer_disc, step_size=20, gamma=0.5)

    loss_discriminator = torch.nn.BCELoss()
    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pkl'))
    def save_model2(name):
        torch.save(model_hal.state_dict(), osp.join(args.save_path, name + '.pkl'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_D_loss'] = []
    trlog['train_dis_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()
        lr_scheduler_disc.step()
        lr_scheduler_h.step()

        model.train()
        model_hal.train()
        model_disc.train()

        tl = Averager()
        ta = Averager()
        tld = Averager()

        for i, batch in enumerate(train_loader, 1):
                       
            data, _ = [_.cuda() for _ in batch]            
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            #print(proto.shape)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)
            hal_i = np.random.randint(args.train_way)
            proto_h = torch.zeros(1600).cuda()
            proto_h_ = torch.zeros(1600).cuda()
            for m in range(args.train_hul):
                
                noise = torch.randn(200).cuda()
                hul_input = torch.cat((proto[hal_i],noise))

                proto_h += model_hal(hul_input)#.unsqueeze(0))
                proto_h_ += model_hal(hul_input)
                #print(proto_h.shape)
            proto_sample = (proto_h + proto[hal_i])/(args.train_hul+1)
            proto_sample2 = (proto_h_ + proto[hal_i])/(args.train_hul+1)
            
            pred_hal = model_disc(proto_sample.detach())
            pred_origin = model_disc(proto[hal_i].detach())
            pred = torch.cat((pred_origin,pred_hal),0)

            lb_hal = torch.zeros(1).cuda()
            lb_origin = torch.ones(1).cuda()
            lb = torch.cat((lb_hal,lb_origin),0)
            
            loss_DiscPlusFt = loss_discriminator(pred,lb)
            
            tld.add(loss_DiscPlusFt.item())
            optimizer_disc.zero_grad()
            loss_DiscPlusFt.backward()
            optimizer_disc.step()


            
            pred_hal = model_disc(proto_sample2.detach())
            optimizer_h.zero_grad()
            loss_h = torch.nn.BCELoss()(pred_hal,lb_origin) #+ 0.1* loss_DiscPlusFt
            loss_h.backward()
            optimizer_h.step()
            

            save_model2('hal')

            proto[hal_i] = (proto_h + proto[hal_i])/(args.train_hul+1)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)           
            q = model(data_query)
            #print(f"proto:{proto.shape}, label:{label.shape}, model(data_query):{q.shape}")
            logits = euclidean_metric(q, proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{},loss_D = {:.4f} loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader),loss_DiscPlusFt.item(), loss.item(), acc),end='\r')
            
            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None
        
        tld = tld.item()
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
        print('\nepoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_dis_loss'].append(tl)
        trlog['train_D_loss'].append(tld)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
    
    plt.figure(figsize=(10,5))
    plt.title("Tr_dis_Loss_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul))
    plt.plot(trlog['train_dis_loss'])
    plt.savefig(osp.join(args.save_path,"Tr_dis_Loss_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul)))

    plt.figure(figsize=(10,5))
    plt.title("Tr_D_Loss_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul))
    plt.plot(trlog['train_D_loss'])
    plt.savefig(osp.join(args.save_path,"Tr_D_Loss_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul)))
    
    plt.figure(figsize=(10,5))
    plt.title("Tr_Acc_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul))
    plt.plot(trlog['train_acc'])
    plt.savefig(osp.join(args.save_path,"Tr_Acc_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul)))
    

    plt.figure(figsize=(10,5))
    plt.title("val_Acc_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul))
    plt.plot(trlog['val_acc'])
    plt.savefig(osp.join(args.save_path,"val_Acc_Ep%d_HM%d.png"%(args.max_epoch,args.train_hul)))
    