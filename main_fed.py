#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=False, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=False, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
        #dict_sever = mnist_spl(dataset_test, 10)

    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=False, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=False, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    qua=[0,0,0,0,0,0,0,0,0,0] #sum quality of client
    good = [0,0,0,0,0,0,0,0,0,0]  # number of good effect
    bad = [0,0,0,0,0,0,0,0,0,0]  # number of bad effect
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users=[0,10,20,30,40,50,60,70,80,90]
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        qua_everyep = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #client_var = []
        i=0
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            clocal=copy.deepcopy(w)
            #print('numclient {:3d}, client loss {}'.format(idx, clocal))

            #client_var=mean(clocal,axis=0)
            #for col in clocal:
             #   tmp=clocal[col].mean()
              #  client_var.append(tmp)
                #print("该列数据的均值位%.5f" % clocal[col].mean())

            loss_locals.append(copy.deepcopy(loss))
            # client testing
            net_glob.load_state_dict(clocal)
            net_glob.eval()
            cacc_train, closs_train = test_img(net_glob, dataset_train, args)
            cacc_test, closs_test = test_img(net_glob, dataset_test, args)
            print("numclient {:3d},Training accuracy: {:.2f}".format(idx,cacc_train))
            print("Testing accuracy: {:.2f}".format(cacc_test))
            qua_everyep[i]=cacc_test.__float__()
            qua[i]+=cacc_test.__float__()
            print(qua[i])
            i=i+1
            #qua.append(cacc_test)

        #compute number of effect every epoch
        sum_qua=0
        for a in range(0, 10):
            sum_qua += qua_everyep[a]
        avgqua_everyep =sum_qua/10
        for a in range(0, 10):
            if qua_everyep[a] > avgqua_everyep:
                good[a] = good[a] + 1
            else:
                bad[a] = bad[a] + 1
            print(good[a],bad[a])

        # update global weights
        w_glob = FedAvg(w_locals)
        #for k, v in w_glob.items():
            #print("每个时隙的均值为：")
            #print (k, v)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # compute every client's quality
    sum = 0
    quav=[0,0,0,0,0,0,0,0,0,0]
    for a in range(0, len(qua)):
        quav[a] = qua[a] / args.epochs
        sum += quav[a]
        print(quav[a])
    avgqua = sum / 10
    print("The average of client's quality: {:.2f}".format(avgqua))

    #reputation-1.effect
    effect=[0,0,0,0,0,0,0,0,0,0]
    b_effect=[0,0,0,0,0,0,0,0,0,0]
    goodw=0.3  #weight of good effect
    badw=0.7   #weight of bad effect
    u=0.1 #fault probability
    y=0.1 #effect probability of fault
    for a in range(0, 10):
        b_effect[a]=(1-u)*goodw*good[a]/(goodw*good[a]+badw*bad[a])
        effect[a]=y*u+b_effect[a]
        print("The reputation of effect for client  {:3d}: {:.8f}".format(idxs_users[a],effect[a]))

    #reputation-2.time
    time_re =[0,0,0,0,0,0,0,0,0,0]
    z=0.9 #fading parameters(0,1)
    Y_time = 5
    #Y_time=10 #time windows
    #y_t=[[1,3,4,7,9],[2,5,8,9],[4,5,7,9],[1,2,4,8],[1,5,6,9],[4,5,6,7,8],[1,5,6],[2,5,8],[3,4,7,9],[2,5,7,8,9]]   #time of event for ever client
    y_t = [[1, 3, 4,], [2], [4], [1, 2, 4], [1], [4], [1],[2], [3, 4], [2]]  # time of event for ever client
    effect_t=[[0.01, 0.91, 0.46,], [0.28], [0.72], [0.01, 0.21, 0.91], [0.91], [0.72], [0.46],[0.28], [0.72, 0.91], [0.72]]  # every time'a effect of event for ever client
    b_time=[0,0,0,0,0,0,0,0,0,0]
    u_time=[0,0,0,0,0,0,0,0,0,0]
    for i in range(0, 10):
        sum_b = 0
        sum_t=0
        sum_u=0
        sum_T=0
        for a in range(0, len(y_t[i])):
            new_level=z**(Y_time-y_t[i][a])
            #print(new_level)
            sum_T=sum_T+new_level*effect_t[i][a]
            #sum_b=sum_b+new_level*b_effect[i]
            sum_u=sum_u+new_level*u
            sum_t=sum_t+new_level
        #b_time[i]=sum_b/sum_t
        #print(b_time[i])
        u_time[i]=sum_u/sum_t
        #print(u_time[i])
        #time_re[i]=y*u_time[i]+b_time[i]
        time_re[i] =sum_T/sum_t
        print("The reputation of time for client  {:3d}: {:.8f}".format(idxs_users[i], time_re[i]))

    # reputation-3.place
    place=[0,0,0,0,0,0,0,0,0,0]
    b_place=[0,0,0,0,0,0,0,0,0,0]
    u_place=[0,0,0,0,0,0,0,0,0,0]
    distance=[0,0,0,0,0,0,0,0,0,0]
    d=[3,5,1,2,4,6,8,1,4,6]
    d_y=0.1 #weight od distance
    avgd=0
    sumd=0
    for i in range(0, 10):
        sumd+=d[i]
    avgd=sumd/10
    for i in range(0, 10):
        distance[i]=d[i]-avgd
        u_place[i]=u_time[i]*(1-d_y)+(1-distance[i]/avgd)*d_y
        place[i]=time_re[i]*(1-d_y)+(1-distance[i]/avgd)*d_y
        b_place[i]=place[i]-y*u_place[i]
        print("The reputation of place for client  {:3d}: {:.8f}".format(idxs_users[i], place[i]))

    # reputation-4.Opinions reputation
    opinion=[0,0,0,0,0,0,0,0,0,0]
    b_opinion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    u_opinion = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    I=[0,10,20,30,40,50,60,70,80,90]#the current sever
    X=[[0,2,34,50,60,70,82,90,91,93],[10,20,50,55,66,77,78,80,90,99],[0,11,20,30,44,55,60,71,80,90],[11,20,33,40,55,66,77,88,90,96],[0,1,10,20,40,50,60,70,77,80]]# the opinion sever on the blockchain
    X_rep = [[0.91,0.01,0.21,0.28,0.46,0.72,0.91,0.21,0.46,0.72],[0.01,0.91,0.28,0.28,0.91,0.72,0.91,0.28,0.91,0.72],[0.72,0.91,0.21,0.46,0.72,0.91,0.01,0.21,0.28,0.46],[0.91,0.01,0.21,0.46,0.72,0.21,0.28,0.46,0.72,0.91],[0.91,0.91,0.21,0.46,0.72,0.01,0.21,0.28,0.46,0.72]]
    IxX=[[0,50,60,70,90],[10,20,50,80,90],[0,20,30,60,80,90],[20,40,90],[0,10,20,40,50,60,70,80]]
    sim_IX=[0,0,0,0,0]
    sum_I=[0,0,0,0,0]
    sum_X=[0,0,0,0,0]
    avg_I=[0,0,0,0,0]
    avg_X=[0,0,0,0,0]
    for j in range(0, 5):
        for i in range(0, 10):
            for a in range(0, len(IxX[j])):
                if I[i]==IxX[j][a]:
                    sum_I[j]=sum_I[j]+place[i]
                if X[j][i]==IxX[j][a]:
                    sum_X[j]=sum_X[j]+X_rep[j][i]
        avg_I[j]=sum_I[j]/len(IxX[j])
        avg_X[j]=sum_X[j]/len(IxX[j])
    sum_t=[0,0,0,0,0]
    t1=[0,0,0,0,0]
    t2=[0,0,0,0,0]
    t3=[0,0,0,0,0]
    for j in range(0, 5):
        for a in range(0, len(IxX[j])):
            for i in range(0, 10):
                if I[i]==IxX[j][a]:
                    t1[j]=place[i]-avg_I[j]
                if X[j][i]==IxX[j][a]:
                    t2[j]=X_rep[j][i]-avg_X[j]
            t3[j]=t1[j]*t2[j]
            sum_t[j]=sum_t[j]+t3[j]
    sum_t1=[0,0,0,0,0]
    sum_t2=[0,0,0,0,0]
    t4=[0,0,0,0,0]
    t5=[0,0,0,0,0]
    for j in range(0, 5):
        for i in range(0, 10):
            t4[j]=(place[i]-avg_I[j])*(place[i]-avg_I[j])
            sum_t1[j]=sum_t1[j]+t4[j]
    for j in range(0, 5):
        for i in range(0, 10):
            t5[j]=(X_rep[j][i]-avg_X[j])*(X_rep[j][i]-avg_X[j])
            sum_t2[j]=sum_t2[j]+t5[j]
        sim_IX[j]=sum_t[j]/(math.sqrt(sum_t1[j])*math.sqrt(sum_t2[j]))
        print(sim_IX[j])
    weight_opinion=0.2
    wIX=[0,0,0,0,0]
    for j in range(0, 5):
        wIX[j]=weight_opinion*sim_IX[j]
    sum_wIX=0
    for j in range(0, 5):
        sum_wIX=sum_wIX+wIX[j]
        print(sum_wIX)
    sum_op=[0,0,0,0,0,0,0,0,0,0]
    for i in range(0,10):
        for j in range(0,5):
            for a in range(0, 10):
                if I[i]==X[j][a]:
                    sum_op[i]=sum_op[i]+X_rep[j][a]*sim_IX[j]
                    print(sum_op[i])
        opinion[i]=sum_op[i]/sum_wIX
        print(opinion[i])
        u_opinion[i]=u
        b_opinion[i]=opinion[i]-y*u_opinion[i]
        print("The reputation of other sever opinion for client  {:3d}: {:.8f}".format(idxs_users[i], opinion[i]))

    # final reputation
    b_final=[0,0,0,0,0,0,0,0,0,0]
    u_final=[0,0,0,0,0,0,0,0,0,0]
    reputation=[0,0,0,0,0,0,0,0,0,0]
    for i in range(0,10):
        b_final[i]=(b_place[i]*u_opinion[i]+b_opinion[i]*u_place[i])/(u_place[i]+u_opinion[i]-u_opinion[i]*u_place[i])
        u_final[i]=(u_place[i]*u_opinion[i])/(u_place[i]+u_opinion[i]-u_opinion[i]*u_place[i])
        reputation[i]=b_final[i]+y*u_final[i]
        print("The final reputation for client  {:3d}: {:.8f}".format(idxs_users[i], reputation[i]))


    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))


