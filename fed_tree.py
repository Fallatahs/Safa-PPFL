#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# same format as chain with reaclling previously trained nodes

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

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    w_locals_all_paths = []

    
    

    for iter in range(args.epochs):
        # print(f"**** Processing iteration {iter + 1}/{args.epochs}")

        w_serial=[]
        w_serial_path = []
        all_paths = []
        all_path_results = []
        Avg_serial_all_paths = []
        avg_w_glob_all_paths = []
        loss_avg_per_path = []
        w_locals_all_paths = []
        trained_nodes = {}  # dictionary to store local weights for each node
        
        #init weight noise (server can remove it in the end of iter)
        for lk in w_glob.keys():
            w_noise[lk] = torch.rand(w_noise[lk].size())

        w_serial.append(copy.deepcopy(w_noise))
        w_noise_plussed = copy.deepcopy(w_noise)

        m = max(int(args.frac * args.num_users), 1) # Randomly select users for the current iteration
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
 
        print("Chain Structure:", idxs_users)
        root = build([int(user_idx) for user_idx in idxs_users])
        # root= build_tree_from_paths(idxs_users)

        print("Binary Tree Structure:")
        print(root)
        
        # 1print("Extracted Paths:")
        get_paths(root, [], all_paths)
        print("Total number of paths:", len(all_paths))
        print("Tree paths are:", all_paths)
        
        w_locals_node = []
        loss_node=[]
        
         # Iterate through all_paths and print each path
        for path_idx, path in enumerate(all_paths, start=1):

            #  print(f"Processing Path {path_idx}: {path}")
            w_locals_path= []
            loss_locals_path = []

                 
            for x in path:
                # Check if the node has been trained before
                if x in trained_nodes:
                    # Recall previous local weights and losses
                    w_locals_node = trained_nodes[x]['weights']
                    loss_node = trained_nodes[x]['loss']                    
                else:
                    # Train the node if it hasn't been trained before
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[x], client_id=x)
                    w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                    
                    w_locals_node = copy.deepcopy(w)
                    loss_node = copy.deepcopy(loss)

                    # Store local weights and losses for the node
                    trained_nodes[x] = {'weights': w_locals_node, 'loss': loss_node}
                    
                w_noise_plus = copy.deepcopy(w_locals_node)   
                for lk in w_noise_plus.keys():
                    w_noise_plussed[lk] += w_noise_plus[lk]

                # print(f"Node {x}: Loss = {loss_node}")

                w_serial.append(copy.deepcopy(w_noise_plussed))
                w_locals_path.append(copy.deepcopy(w_locals_node))
                loss_locals_path.append(copy.deepcopy(loss_node))
                
            w_locals_all_paths.append(copy.deepcopy(w_locals_path))
          
            Avg_serial_path = FedAvg_serial(w_serial)
            Avg_serial_all_paths.append(copy.deepcopy(Avg_serial_path))
        
            loss_avg_path = sum(loss_locals_path) / len(loss_locals_path)
            loss_avg_per_path.append(loss_avg_path)
            print('Round {:3d}, Path {:3d} Average loss {:.3f}'.format(iter, path_idx, loss_avg_path))
        
        # for node, node_results in trained_nodes.items():
        #     # print(f"Node {node} Loss Values: {node_results['loss']}")
        #     # plt.plot(node_results['loss'], label=f'Node {node}')
        #     loss_values = node_results['loss']
        #     print(f'Node {node} Final Loss: {loss_values}')
            
        # loss_values = [node_results['loss'] for node_results in trained_nodes.values()]
        # nodes = list(trained_nodes.keys())
        # plt.bar(nodes, loss_values)
        # plt.xlabel('Node')
        # plt.ylabel('Final Loss')
        # plt.show()
        

        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig('loss_plot.png')

        w_glob = FedAvg(Avg_serial_all_paths)
        net_glob.load_state_dict(w_glob)

        # print("in iter len loss_avg_per_path", len(loss_avg_per_path))
        tree_loss_avg = sum(loss_avg_per_path) / len(loss_avg_per_path)
        print('Round {:3d}, Total average loss {:.3f}'.format(iter, tree_loss_avg))
        print("*************")
        loss_train.append(tree_loss_avg)
        # print("in iter len loss_train.append", len(loss_train))
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time} seconds")

    lossfile = open('./log/lostfile_{}_{}_{}_iid{}_tree.dat'.format(args.dataset, args.model, args.epochs, args.iid), "w")

    for lo in loss_train:
        slo = str(lo)
        lossfile.write(slo)
        lossfile.write('\n')
    lossfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./log/fed_tree_rand_{}_{}_{}_C{}_iid{}_tree.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing

    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    

