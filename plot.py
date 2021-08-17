import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
from networkx.algorithms.cycles import *
import matplotlib.pyplot as plt


# loadData = np.load('mnist_to_graph/graphs/0.npy')
# print(loadData)


class GetGraph:

    def __init__(self):
        pass

    @staticmethod
    def create_directed_graph(data):
        my_graph = nx.DiGraph()
        my_graph.clear()
        # for front_node, back_node_list in data:
        #     if back_node_list:
        #         for back_node in back_node_list:
        #             my_graph.add_edge(front_node, back_node)
        #     else:
        #         my_graph.add_node(front_node)

        for relationship in data:
            my_graph.add_edge(relationship[0], relationship[1])

        return my_graph

    @staticmethod
    def create_undirected_graph(data):
        my_graph = nx.Graph()
        my_graph.clear()
        # for front_node, back_node_list in data:
        #     if back_node_list:
        #         for back_node in back_node_list:
        #             my_graph.add_edge(front_node, back_node)
        #     else:
        #         my_graph.add_node(front_node)

        for relationship in data:
            my_graph.add_edge(relationship[0], relationship[1])

        return my_graph

    @staticmethod
    def draw_directed_graph(my_graph, name='out_directed_graph'):
        nx.draw_networkx(my_graph, pos=nx.circular_layout(my_graph), vmin=10,
                         vmax=20, width=2, font_size=8, edge_color='black')
        picture_name = name + ".png"
        plt.savefig(picture_name)
        # print('save success: ', picture_name)
        # plt.show()

    @staticmethod
    def draw_undirected_graph(my_graph, name='out_undirected_graph'):
        nx.draw_networkx(my_graph, pos=nx.random_layout(my_graph), vmin=10,
                         vmax=20, width=2, font_size=8, edge_color='black')
        picture_name = name + ".png"
        plt.savefig(picture_name)
        print('save success: ', picture_name)
        # plt.show()

    @staticmethod
    def get_next_node(my_graph):
        nodes = my_graph.nodes
        next_node_dict = {}
        for n in nodes:
            value_list = list(my_graph.successors(n))
            next_node_dict[n] = value_list
        return copy.deepcopy(next_node_dict)

    @staticmethod
    def get_front_node(my_graph):
        nodes = my_graph.nodes
        front_node_dict = {}
        for n in nodes:
            value_list = list(my_graph.predecessors(n))
            front_node_dict[n] = value_list
        return copy.deepcopy(front_node_dict)

    @staticmethod
    def get_loop_node(my_graph):
        loop = (list(simple_cycles(my_graph)))
        return copy.deepcopy(loop)


if __name__ == '__main__':
    comp_graph_object = GetGraph()
    features = np.load('mnist_to_graph/node_features_knn/1000.npy')
    print("*********************************************************")
    # print(features)

    comp_statement = []

    for feature in features:
        first = feature[0]
        second = feature[1]
        # comp_statement[first] = [f'{second}']
        comp_statement.append((first, second))
    # print(comp_statement)
    # print('self.comp_statement_ct_map:', comp_statement)
    graph = comp_graph_object.create_undirected_graph(comp_statement)
    # comp_next_node = comp_graph_object.get_next_node(graph)
    # comp_front_node = comp_graph_object.get_front_node(graph)
    # comp_loop_list = comp_graph_object.get_loop_node(graph)
    comp_graph_object.draw_undirected_graph(graph)

    # plt.tight_layout()
    # plt.show()
    # plt.savefig("Graph.png", format="PNG")




