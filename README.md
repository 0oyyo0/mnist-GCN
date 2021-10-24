#### 数据准备过程
如何使用GCN去进行手写数字的识别呢？首先把手写数字图像转换成图数据才行，得到的结果就是一张手写数字的图片转换为两个数据，node和edge。
+ 构建图神经网络所需要的Data数据
```python
data_list = []
for i in range(data_size):
    edge = torch.tensor(np.load('./dataset/graphs/'+str(i)+'.npy').T,dtype=torch.long)
    x = torch.tensor(np.load('./dataset/node_features/'+str(i)+'.npy')/28,dtype=torch.float) 

    d = Data(x=x, edge_index=edge.contiguous(),t=int(labels[i]))
    data_list.append(d)
```
data_list中每个元素都是Data，Data里面又包含x、t（target）、edge_index等图基本信息。 其中可以看到x除以28是为了将feature的值转为0-1之间。

+ 有了这个data_list，便可以直接输入到Dataloader里面去了。

```python
trainloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
```
这样就构建好dataloder了～

#### 构建GCN网络
```python
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(2, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 48)
        self.conv4 = GCNConv(48, 64)
        self.conv5 = GCNConv(64, 96)
        self.conv6 = GCNConv(96, 128)
        self.linear1 = torch.nn.Linear(128,64)
        self.linear2 = torch.nn.Linear(64,10)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
    
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)
        x = F.relu(x)
        x, _ = scatter_max(x, data.batch, dim=0)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
```
然后训练过程跟普通的网络没啥区别