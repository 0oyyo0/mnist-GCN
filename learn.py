import numpy as np



# features = np.load('mnist_to_graph/node_features/184.npy')
# print(features.shape)
# print(features)

# print("*********************************************************")

# features2 = np.load('mnist_to_graph/node_features2/184.npy')
# print(features2.shape)
# print(features2)
# print(features2 == features)



edges = np.load('mnist_to_graph/graphs/187.npy')
print(edges.shape)
print(edges)
print("*********************************************************")

edges2 = np.load('mnist_to_graph/graphs_knn/189.npy')
print(edges2.shape)
print(edges2)

print(edges == edges2)