from random import random

import numpy
import pandas


class Node:
    error = 0
    name = ""
    from_edges = None
    to_edges = None

    def __init__(self, name):
        self.name = name
        self.error = 0
        self.from_edges = list()
        self.to_edges = list()

    def calculate_output(self):
        edges = self.to_edges
        sum = 0
        for edge in edges:
            sum = sum + (edge.weight * edge.value)
        return sigmoid(sum)

    def calculate_input_error(self):
        edges = self.from_edges
        error = edges[0].value * (1 - edges[0].value)
        my_sum = 0
        for edge in edges:
            to_node = edge.to_node
            my_sum = my_sum + (edge.weight * to_node.error)
        self.error = error * my_sum

    def calculate_output_error(self, output, target):
        self.error = output * (1 - output) * (target - output)


class Edge:
    from_node = None
    to_node = None
    weight = 0
    value = 0

    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.value = 0

    def is_from_node(self, node_name):
        if self.from_node.name == node_name:
            return True
        return False

    def is_to_node(self, node_name):
        if self.to_node.name == node_name:
            return True
        return False


def main():
    number_of_node_for_each_layer = convert_txt_to_csv_training("train.txt", "train.csv")
    column_name = list()
    for i in range(0, number_of_node_for_each_layer[0]):
        column_name.append("I" + str(i))
    for i in range(0, number_of_node_for_each_layer[2]):
        column_name.append("O" + str(i))
    dataset = pandas.read_csv("train.csv", names=column_name)
    dataset[dataset.columns[number_of_node_for_each_layer[0]:]] = (dataset[dataset.columns[
                                                                           number_of_node_for_each_layer[0]:]] -
                                                                   dataset[dataset.columns[
                                                                           number_of_node_for_each_layer[
                                                                               0]:]].min()) / (dataset[dataset.columns[
                                                                                                       number_of_node_for_each_layer[
                                                                                                           0]:]].max() -
                                                                                               dataset[dataset.columns[
                                                                                                       number_of_node_for_each_layer[
                                                                                                           0]:]].min())
    dataset[dataset.columns[:number_of_node_for_each_layer[0]]] = (dataset[dataset.columns[
                                                                           :number_of_node_for_each_layer[0]]] -
                                                                   dataset[dataset.columns[
                                                                           :number_of_node_for_each_layer[0]]].mean()) / \
                                                                  dataset[dataset.columns[
                                                                          :number_of_node_for_each_layer[0]]].std()
    input_layer = list()
    hidden_layer = list()
    output_layer = list()
    intialize_input_layer_nodes(column_name, input_layer, number_of_node_for_each_layer)
    intialize_hidden_layer_nodes(hidden_layer, number_of_node_for_each_layer)
    initialize_output_layer_nodes(column_name, number_of_node_for_each_layer, output_layer)
    create_network(hidden_layer, input_layer, output_layer)
    start_training(dataset, input_layer, hidden_layer, output_layer)
    convert_txt_to_csv_test("test.txt", "test.csv")
    test_set = pandas.read_csv("test.csv")
    test_set = (test_set - test_set.mean()) / test_set.std()
    start_testing(test_set, input_layer, hidden_layer, output_layer)


def start_testing(test_set, input_layer, hidden_layer, output_layer):
    mse = 0
    sum = 0
    sum_of_mse = 0
    for i in range(0, len(test_set)):
        row = test_set.iloc[i]
        counter = 0
        for i_node in input_layer:
            edges = i_node.from_edges  # get_edge_by_to(i_node.name, list_edges)
            for edge in edges:
                edge.value = row.iloc[counter]
            counter += 1
        for h_node in hidden_layer:
            output = h_node.calculate_output()
            edges = h_node.from_edges  # get_edge_by_to(h_node.name, list_edges)
            for edge in edges:
                edge.value = output
        for o_node in output_layer:
            output = o_node.calculate_output()
            sum = sum + numpy.power((row.iloc[counter] - output), 2)
            counter += 1
            mse = sum / 2
            sum_of_mse += mse
    sum_of_mse = sum_of_mse / len(test_set)
    print(f"mean square error {sum_of_mse}")


def start_training(dataset, input_layer, hidden_layer, output_layer):
    iterations = 0
    number_of_iterations = 500
    mse = 0
    learning_rate = 0.05
    while True:
        sum = 0
        sum_of_mse = 0
        for i in range(0, len(dataset)):
            row = dataset.iloc[i]
            counter = 0
            for i_node in input_layer:
                edges = i_node.from_edges  # get_edge_by_to(i_node.name, list_edges)
                for edge in edges:
                    edge.value = row.iloc[counter]
                counter += 1
            for h_node in hidden_layer:
                output = h_node.calculate_output()
                edges = h_node.from_edges  # get_edge_by_to(h_node.name, list_edges)
                for edge in edges:
                    edge.value = output
            for o_node in output_layer:
                output = o_node.calculate_output()
                sum = sum + numpy.power((row.iloc[counter] - output), 2)
                counter += 1
                mse = sum / 2
                sum_of_mse += mse
        sum_of_mse = sum_of_mse / len(dataset)
        print(f"mean square error {sum_of_mse}")
        iterations += 1
        if iterations == number_of_iterations:
            break
        for i in range(0, len(dataset)):
            row = dataset.iloc[i]
            counter = 0
            for i_node in input_layer:
                edges = i_node.from_edges  # get_edge_by_to(i_node.name, list_edges)
                for edge in edges:
                    edge.value = row.iloc[counter]
                counter += 1
            for h_node in hidden_layer:
                output = h_node.calculate_output()
                edges = h_node.from_edges  # get_edge_by_to(h_node.name, list_edges)
                for edge in edges:
                    edge.value = output
            for o_node in output_layer:
                output = o_node.calculate_output()
                o_node.calculate_output_error(output, row.iloc[counter])
                edges = o_node.to_edges
                for edge in edges:
                    edge.weight = edge.weight + (learning_rate * o_node.error * edge.value)
                counter += 1
            for h_node in hidden_layer:
                edges = h_node.to_edges  # get_edge_by_from(h_node.name, list_edges)
                h_node.calculate_input_error()
                for edge in edges:
                    edge.weight = edge.weight + (learning_rate * h_node.error * edge.value)


def create_network(hidden_layer, input_layer, output_layer):
    for node in input_layer:
        for h_node in hidden_layer:
            edge = Edge(node, h_node, random())
            node.from_edges.append(edge)
            h_node.to_edges.append(edge)
    for h_node in hidden_layer:
        for o_node in output_layer:
            edge = Edge(h_node, o_node, random())
            h_node.from_edges.append(edge)
            o_node.to_edges.append(edge)


def initialize_output_layer_nodes(column_name, number_of_node_for_each_layer, output_layer):
    for i in range(number_of_node_for_each_layer[0], len(column_name)):
        output_layer.append(Node(column_name[i]))


def intialize_hidden_layer_nodes(hidden_layer, number_of_node_for_each_layer):
    for i in range(0, number_of_node_for_each_layer[1]):
        hidden_layer.append(Node("H" + str(i)))


def intialize_input_layer_nodes(column_name, input_layer, number_of_node_for_each_layer):
    for i in range(0, number_of_node_for_each_layer[0]):
        input_layer.append(Node(column_name[i]))


def convert_txt_to_csv_training(txt_name, csv_name):
    file = open(txt_name, "r+")
    output_file = open(csv_name, "w")
    number_of_node_for_each_layer = list()
    counter = 0
    for line in file:
        line = line.strip()
        if counter == 0:
            values = line.split(" ")
            new_line = get_new_line(values)
            new_line_values = new_line.split(",")
            for i in range(0, 3):
                number_of_node_for_each_layer.append(int(new_line_values[i]))
        elif counter > 1:
            values = line.split(" ")
            new_line = get_new_line(values)
            output_file.write(new_line + "\n")
        counter += 1
    file.close()
    output_file.close()
    return number_of_node_for_each_layer


def convert_txt_to_csv_test(txt_name, csv_name):
    file = open(txt_name, "r+")
    output_file = open(csv_name, "w")
    for line in file:
        line = line.strip()
        values = line.split(" ")
        new_line = get_new_line(values)
        output_file.write(new_line + "\n")
    file.close()
    output_file.close()


def get_new_line(values):
    new_line = ""
    for i in range(0, len(values)):
        if values[i] != "":
            new_line = new_line + values[i]
            if i < len(values) - 1:
                new_line = new_line + ","
    return new_line


def get_edge_by_from(node_name, list_edges):
    edges = list()
    for edge in list_edges:
        if edge.is_from_node(node_name):
            edges.append(edge)
    return edges


def get_edge_by_to(node_name, list_edges):
    edges = list()
    for edge in list_edges:
        if edge.is_to_node(node_name):
            edges.append(edge)
    return edges


def sigmoid(value):
    return 1 / (1 + numpy.exp(-value))


main()
