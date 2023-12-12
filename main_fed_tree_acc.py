#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import time

from binarytree import build, Node, tree
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg, FedAvg_serial
from models.test import test_img

# =========================================================================

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(values):
    if len(values) == 0:
        return None

    root = Node(values[0])
    queue = [root]
    i = 1

    while i < len(values):
        current = queue.pop(0)

        if values[i] is not None:
            current.left = Node(values[i])
            queue.append(current.left)
        i += 1

        if i < len(values) and values[i] is not None:
            current.right = Node(values[i])
            queue.append(current.right)
        i += 1

    return root

def get_paths(node, current_path, all_paths):
    if node is None:
        return

    # 1: Append the current node's value to the current path
    current_path.append(node.value)

    # 2: Check if the current node is a leaf node
    if node.left is None and node.right is None:
        # Leaf node, add the current path to all_paths
        all_paths.append(current_path.copy())
    else:
        # 3: Recursively traverse left and right
        get_paths(node.left, current_path, all_paths)
        # printPathsRec(root.left, path, pathLen)
        get_paths(node.right, current_path, all_paths)

    # 4. Backtrack by removing the current node from the current path
    current_path.pop() #backtrack step to explore other paths
# =========================================================================    

# def print_tree(root):
#     if not root:
#         return

#     queue = [root]

#     while queue:
#         node = queue.pop(0)
#         print(node.value, end=' ')

#         if node.left:
#             queue.append(node.left)
#         if node.right:
#             queue.append(node.right)
# =========================================================================
# class TreeNode:
#     def __init__(self, value):
#         self.value = value
#         self.left = None
#         self.right = None
# def build_tree_from_paths(paths):
#     if not paths:
#         return None

#     root = TreeNode(paths[0][0])
#     for path in paths:
#         current = root
#         for value in path[1:]:
#             if value not in [node.value for node in current.children()]:
#                 new_node = TreeNode(value)
#                 if current.left is None:
#                     current.left = new_node
#                 else:
#                     current.right = new_node
#                 current = new_node
#             else:
#                 current = next(node for node in current.children() if node.value == value)

#     return root

# def print_tree(root):
#     if not root:
#         return

#     queue = [root]

#     while queue:
#         node = queue.pop(0)
#         print(node.value, end=' ')

#         if node.left:
#             queue.append(node.left)
#         if node.right:
#             queue.append(node.right)

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cpu')
    start_time = time.time()

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
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
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    w_noise = copy.deepcopy(w_glob)

    # for key in w_glob.keys():
    #     print("key=",w_glob[key],"size=",w_glob[key].size())
    
    # os.system("pause")

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
   
    acc_test = []


    

    for iter in range(args.epochs):
        w_serial = []
        w_serial_path = []
        all_paths = []
        all_path_results = []
        Avg_serial_all_paths = []
        avg_w_glob_all_paths = []
        loss_avg_per_path = []
        trained_nodes = {}
       
        w_locals_all_paths = []
        loss_locals_all_paths = []
        acc_test_per_path = []# list to store accuracy for each path in the current round

        #init weight noise (server can remove it in the end of iter)
        for lk in w_glob.keys():
            w_noise[lk] = torch.rand(w_noise[lk].size())

        w_serial.append(copy.deepcopy(w_noise))

        w_noise_plussed = copy.deepcopy(w_noise)

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        root = build([int(user_idx) for user_idx in idxs_users])
        get_paths(root, [], all_paths)
        # print("Total number of paths:", len(all_paths))
        # print("Tree paths are:", all_paths)

        for path_idx, path in enumerate(all_paths, start=1):
            w_locals_path= []
            loss_locals_path = []

            for idx in path:
                if idx in trained_nodes:
                    w_locals_node = trained_nodes[idx]['weights']
                    loss_node = trained_nodes[idx]['loss']                    
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], client_id=idx)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))

                    w_locals_node = copy.deepcopy(w)
                    loss_node = copy.deepcopy(loss)

                    trained_nodes[idx] = {'weights': w_locals_node, 'loss': loss_node}
                
                #  serialy accumulate weight and bias 
                w_noise_plus = copy.deepcopy(w)
                for lk in w_noise_plus.keys():
                    w_noise_plussed[lk] += w_noise_plus[lk]

                w_serial.append(copy.deepcopy(w_noise_plussed))
                w_locals_path.append(copy.deepcopy(w_locals_node))
                loss_locals_path.append(copy.deepcopy(loss_node))
                # print ("Path", path_idx, "len w_locals_path=", len(w_locals_path), "len loss_locals_path=", len(loss_locals_path) )
                
            w_locals_all_paths.append(copy.deepcopy(w_locals_path))   
            loss_locals_all_paths.append(copy.deepcopy(loss_locals_path))
            Avg_serial_path = FedAvg_serial(w_serial)
            Avg_serial_all_paths.append(copy.deepcopy(Avg_serial_path))
            # print ("len loss_locals_all_paths=", len(loss_locals_all_paths))
            
        
            
            # loss_avg_path = sum(loss_locals_path) / len(loss_locals_path)
            # loss_avg_per_path.append(loss_avg_path)

        # update global weights
        w_glob = FedAvg(Avg_serial_all_paths)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # Evaluate the model on the test dataset after each path
        net_glob.eval()

        acc_t, loss_t = test_img(net_glob, dataset_test, args)

        print("Round {:3d}, Num of Chains: {:3d}, Testing accuracy: {:.2f}".format(iter, len(all_paths), acc_t))
       
        
        #print("Training accuracy: {:.2f}".format(acc_train))
        # acc_test_per_path.append(acc_t)#contain the testing accuracy after each path
        # print("Round {:3d}, Path {:3d}, Testing accuracy: {:.2f}".format(iter,path_idx, acc_t))
        
        # Append the last testing accuracy after all paths in the current iteration
        loss_train.append(loss_t) #store the last loss after all paths in each iteration
        acc_test.append(acc_t)#contain the testing accuracy after each iteration

    accfile = open('./log/accfile_{}_{}_{}_iid{}_serial.dat'.format(args.dataset, args.model, args.epochs, args.iid), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig('./log/fed_{}_{}_{}_C{}_iid{}_acc_serial.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing

