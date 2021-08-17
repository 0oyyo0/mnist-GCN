import gzip
import numpy as np
import os.path as osp
import os
from tqdm import tqdm
from sklearn.neighbors import kneighbors_graph

'''
def method_1():
    data = 0

    if not osp.exists(graphs_path):
        os.mkdir(graphs_path)
    if not osp.exists(node_features_path):
        os.mkdir(node_features_path)

    #将MNIST数据做成二维形式
    with gzip.open(mnist_dataset, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16) # [0 0 0 ... 0 0 0]
        data = data.reshape([-1, 28, 28])
    # 找到像素值大于0.4的像素点，作为图数据中的node，小于0.4的像素点去掉
    data = np.where(data < 102, -1, 1000)  #255*0.4

    print("making>>>>")
    for e,imgtmp in tqdm(enumerate(data)):
        img = np.pad(imgtmp, [(2, 2), (2, 2)], "constant", constant_values=(-1))
        cnt = 0
        # 一整张图像中，图node的像素点从0到N进行标号，其实就是表示N个node。
        for i in range(2, 30):
            for j in range(2, 30):
                if img[i][j] == 1000:
                    img[i][j] = cnt
                    cnt+=1
        
        edges = []
        # y座標、x座標
        # node有了，node的feature其实就是这个node的x和y坐标。一个node两个feature。
        features = np.zeros((cnt, 2))
        # 然后就需要边信息，边具体如何找呢，就是对所有node进行循环，然后去找每个node周围的8个像素点，
        for i in range(2, 30):
            for j in range(2, 30):
                if img[i][j] == -1:
                    continue

                #8近傍に該当する部分を抜き取る。
                filter = img[i-2:i+3, j-2:j+3].flatten() # 返回一个一维数组
                filter1 = filter[[6, 7, 8, 11, 13, 16, 17, 18]] # i,j周围的八个像素点
 
                features[filter[12]][0] = i-2  # filter[12]是该像素点
                features[filter[12]][1] = j-2
                # 如果周围的像素点也是node（像素点值大于0.4），那么就直接连通上，如果不是node（即像素点值小于0.4），
                # 那么就不进行连通
                for neighbor in filter1:
                    if not neighbor == -1:
                        edges.append([filter[12], neighbor])

        np.save(graphs_path+str(e), edges)
        np.save(node_features_path+str(e), features)

        # 每张28*28的灰度图都做成了一个graph
        # 训练集一共6000个graph
'''



def knn_adjacent_mat():
    data = 0

    if not osp.exists(graphs_path):
        os.mkdir(graphs_path)
    if not osp.exists(node_features_path):
        os.mkdir(node_features_path)

    #将MNIST数据做成二维形式
    with gzip.open(mnist_dataset, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16) # [0 0 0 ... 0 0 0]
        data = data.reshape([-1, 28, 28])

    # data = np.where(data < 30, 0, data)//正在测试中
    data = np.where(data < 30, 0, 1000) 

    print("making>>>>")
    for name, img in tqdm(enumerate(data)):
        A = kneighbors_graph(img, n_neighbors=9, include_self=True).toarray()
        # print(A)

        edges = []
        edges.append(np.argwhere(A))

        #feature是结点的横纵坐标
        feature = []
        feature.append(np.argwhere(img != -1))

        feature = np.array(feature)
        edges = np.array(edges)

        feature = feature.squeeze()
        edges = edges.squeeze()

        # 只有一个特征，不用归一化
        np.save(graphs_path+str(name), edges)
        np.save(node_features_path+str(name), feature)


if __name__=="__main__":
    mnist_dataset = 'dataset_raw/train-images-idx3-ubyte.gz'
    graphs_path = "mnist_to_graph/graphs_knn/"
    node_features_path = "mnist_to_graph/node_features_knn/"
    # method_1()
    knn_adjacent_mat()