import random
import time
import argparse

import torch.nn as nn
import numpy as np
import torch
from torch.nn import parameter
import torch.optim as optim
from utils import accuracy
from models import GCN
from utils import load_data
from load_graph import load_mnist_graph
from torch_geometric.data import Data,DataLoader



data_size = 60000
train_size = 50000
batch_size = 100
epoch_num = 150

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cora",
                    help='dataset for training')
parser.add_argument('--times', type=int, default=1,
                    help='times of repeat training')
parser.add_argument('--seed', type=int, default=33, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
criterion = torch.nn.NLLLoss()
device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')


def train(epoch, model, optimizer, adj, features, labels, idx_train, idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad() # 清空过往梯度；如果不清零会累加

    output = model(features, adj)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward() # 反向传播，计算当前梯度；查看，例如：y=kx+b y.backward() b.grad x.grad
    optimizer.step() # 根据梯度更新网络参数

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        # 输入的是整个图？
        loss_val = criterion(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

    print(f'Epoch: {epoch + 1:04d}',
          f'loss_train: {loss_train.item():.4f}',
          f'acc_train: {acc_train:.4f}',
          f'loss_val: {loss_val.item():.4f}',
          f'acc_val: {acc_val:.4f}',
          f'time: {time.time() - t:.4f}s')
    return loss_val


@torch.no_grad()
def test(model, adj, features, labels, idx_test):
    model.eval()
    output = model(features, adj) # 输入的是整个图？
    loss_test = criterion(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print(f"Test set results:",
          f"loss= {loss_test.item():.4f}",
          f"accuracy= {acc_test:.4f}")
    return acc_test


def main(dataset, times):
    # adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset)
    # features = features.to(device)
    # adj = adj.to(device)
    # labels = labels.to(device)
    # idx_train = idx_train.to(device)
    # idx_val = idx_val.to(device)
    # idx_test = idx_test.to(device)

    # nclass = labels.max().item() + 1
    mnist_list = load_mnist_graph(data_size=data_size)
    device = torch.device('cuda')
    model = GCN(nfeat=2,
                nhid=args.hidden,
                nclass=10,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
    # for name, parameters in model.named_parameters():
    #     print(name, parameters, parameters.size())

    model.to(device)
    trainset = mnist_list[:train_size]
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = mnist_list[train_size:]
    testloader = DataLoader(testset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss()
    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    print("Start Train")
    
    #学習部分
    model.train()
    for epoch in range(epoch_num):
        train_loss = 0.0
        for i, batch in enumerate(trainloader):
            batch = batch.to("cuda")
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs,batch.t)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.cpu().item()
            if i % 10 == 9:
                progress_bar = '['+('='*((i+1)//10))+(' '*((train_size//100-(i+1))//10))+']'
                print('\repoch: {:d} loss: {:.3f}  {}'
                        .format(epoch + 1, loss.cpu().item(), progress_bar), end="  ")

        print('\repoch: {:d} loss: {:.3f}'
            .format(epoch + 1, train_loss / (train_size / batch_size)), end="  ")
        history["train_loss"].append(train_loss / (train_size / batch_size))

        correct = 0
        total = 0
        batch_num = 0
        loss = 0
        with torch.no_grad():
            for data in testloader:
                data = data.to(device)
                outputs = model(data)
                loss += criterion(outputs,data.t)
                _, predicted = torch.max(outputs, 1)
                total += data.t.size(0)
                batch_num += 1
                correct += (predicted == data.t).sum().cpu().item()

        history["test_acc"].append(correct/total)
        history["test_loss"].append(loss.cpu().item()/batch_num)
        endstr = ' '*max(1,(train_size//1000-39))+"\n"
        print('Test Accuracy: {:.2f} %%'.format(100 * float(correct/total)), end='  ')
        print(f'Test Loss: {loss.cpu().item()/batch_num:.3f}',end=endstr)


    print('Finished Training')

    #最終結果出力
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += data.t.size(0)
            correct += (predicted == data.t).sum().cpu().item()
    print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))



if __name__ == '__main__':
    main(dataset=args.dataset, times=args.times)
