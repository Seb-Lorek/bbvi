"""
Directed Graph.
"""

import networkx as nx

import matplotlib.pyplot as plt

import re

from typing import Any

from .model import (
    Model,
    Lpred,
    Param,
)

class ModelGraph:
    def __init__(self, Model: Model):
        self.Graph = nx.DiGraph()
        self.Model = Model

    def add_root(self) -> None:
        value = {"value": self.Model.y,
                 "dist": self.Model.y_dist.get_dist()}
        self.Graph.add_node("response", value=value)
        self.Graph.nodes["response"]["node_type"] = "strong"

    def add_strong_node(self, name: str, input: Any) -> None:
        value = {"dim": input.dim,
                 "bijector": input.function,
                 "dist": input.distribution.init_dist()}
        self.Graph.add_node(name, value=value)
        self.Graph.nodes[name]["node_type"] = "strong"

    def add_weak_node(self, name: str, input: Any) -> None:
        value = {"bijector": input.function}
        self.Graph.add_node(name, value=value)
        self.Graph.nodes[name]["node_type"] = "weak"

    def add_fixed_node(self, name: str, input:Any) -> None:
        value = {"fixed": input}
        self.Graph.add_node(name, value=value)
        self.Graph.nodes[name]["node_type"] = "weak"

    def build_graph(self) -> None:

        self.add_root()

        # Build graph by searching through the classes
        for kw1, input1 in self.Model.y_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                self.add_weak_node(name=kw1, input=input1)
                self.Graph.add_edge(kw1, "response", role="lpred", bijector=input1.function)
                name = input1.Obs.name
                self.add_fixed_node(name=name, input=input1.design_matrix)
                self.Graph.add_edge(name, kw1, role="fixed")
                for kw2, input2 in input1.kwinputs.items():
                    self.add_strong_node(name=kw2, input=input2)
                    self.Graph.add_edge(kw2, kw1, role="param")
                    for kw3, input3 in input2.distribution.kwinputs.items():
                        name = input3.name
                        self.add_fixed_node(name=name, input=input3.value)
                        self.Graph.add_edge(name, kw2, role="hyper_param")
            elif isinstance(input1, Param):
                self.add_strong_node(name=kw1, input=input1)
                self.Graph.add_edge(kw1, "response", role="param")
                for kw3, input3 in input1.distribution.kwinputs.items():
                    name = input3.name
                    self.add_fixed_node(name=name, input=input3.value)
                    self.Graph.add_edge(name, kw1, role="hyper_param")

    def visualize_graph(self):
        pos = nx.nx_agraph.graphviz_layout(self.Graph, prog="dot")
        nx.set_node_attributes(self.Graph, pos, "pos")

        for node in self.Graph.nodes:
            self.Graph.nodes[node]["label"] = f"{node}"

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Model Graph")
        ax.set_axis_off()

        pos = nx.get_node_attributes(self.Graph, "pos")
        labels = nx.get_node_attributes(self.Graph, "label")
        node_type = nx.get_node_attributes(self.Graph, "node_type")

        node_shapes = {
            "strong": "o",
            "weak": ""
        }

        node_labels = {node: label for node, label in labels.items() if node in node_type}

        strong_nodes = [node for node, node_type in node_type.items() if node_type == "strong"]
        weak_nodes = [node for node, node_type in node_type.items() if node_type == "weak"]

        nx.draw_networkx_edges(self.Graph, pos, ax=ax, node_size=1000)
        nx.draw_networkx_labels(self.Graph, pos, ax=ax, labels=node_labels, font_size=10)

        nx.draw_networkx_nodes(self.Graph, pos, nodelist=strong_nodes, ax=ax, node_color="lightblue",
                               node_shape=node_shapes["strong"], node_size=1000)
        nx.draw_networkx_nodes(self.Graph, pos, nodelist=weak_nodes, ax=ax, node_color="lightblue",
                               node_shape=node_shapes["weak"], node_size=1000)

        edge_labels = nx.get_edge_attributes(self.Graph, "bijector")

        for key, value in edge_labels.items():
            if value is None:
                edge_labels[key] = "identity"
            else:
                string_to_search = str(edge_labels[key])
                pattern = r"jax\.numpy\.(\w+)"
                match_pattern = re.search(pattern, string_to_search)
                edge_labels[key] = match_pattern.group()

        nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels=edge_labels, ax=ax, font_size=8)

        plt.show()
