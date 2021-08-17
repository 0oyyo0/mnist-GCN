import numpy as np
import gzip
import torch
from torch_geometric.data import Data

def load_mnist_graph(data_size=60000):
    data_list = []
    labels = 0
    with gzip.open('./dataset_raw/train-labels-idx1-ubyte.gz', 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
        
    for i in range(data_size):
        edge = torch.tensor(np.load('./mnist_to_graph/graphs_knn/'+str(i)+'.npy').T,dtype=torch.long)
        x = torch.tensor(np.load('./mnist_to_graph/node_features_knn/'+str(i)+'.npy')/28,dtype=torch.float) 

        d = Data(x=x, edge_index=edge.contiguous(),t=int(labels[i]))
        data_list.append(d)
        if i%1000 == 999:
            print("\rData loaded "+ str(i+1), end="  ")

    print("Complete!")
    return data_list


if __name__ == "__main__":
    data_list = load_mnist_graph(6000)
    print(data_list)