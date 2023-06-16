"""
Directed Graph.
"""

import jax.numpy as jnp

import networkx as nx
import matplotlib.pyplot as plt
import re

from typing import Any

from .utils import (
    dot
)

from .model import (
    Model,
    Lpred,
    Param,
    Array,
    Distribution
)

class ModelGraph:
    def __init__(self, Model: Model):
        self.DiGraph = nx.DiGraph()
        self.Model = Model
        self.traversal_order = None
        self.update_traversal_order = []
        self.prob_traversal_order = []

    def add_root(self) -> None:
        """
        Method to generate the root of the DAG (response).
        """

        attr = {"value": self.Model.response,
                 "dist": self.Model.response_dist.get_dist(),
                 "log_lik": self.Model.log_lik}
        self.DiGraph.add_node("response", attr=attr)
        self.DiGraph.nodes["response"]["node_type"] = "root"
        self.DiGraph.nodes["response"]["input"] = {}

    def add_strong_node(self, name: str, input: Any) -> None:
        """
        Method to add a strong node to the DAG. Strong nodes have a probability distribution.

        Args:
            name (str): Name of the node. Key of kwinputs.
            input (Any): Class param. Value of kwinputs.
        """

        attr = {"dim": input.dim,
                 "value": input.value,
                 "bijector": input.function,
                 "dist": input.distribution.get_dist(),
                 "log_prior": input.log_prob}
        self.DiGraph.add_node(name, attr=attr)
        self.DiGraph.nodes[name]["node_type"] = "strong"
        self.DiGraph.nodes[name]["input"] = {}

    def add_lpred_node(self, name: str, input: Any) -> None:
        """
        Method to add a linear predictor (lpred) node to the DAG. Lpred nodes are linear predictors
        with a bijector function (inverse link function).

        Args:
            name (str): Name of the weak node.
            input (Any): Jax.numpy bijector.
        """

        attr = {"value": input.value,
                 "bijector": input.function}
        self.DiGraph.add_node(name, attr=attr)
        self.DiGraph.nodes[name]["node_type"] = "lpred"
        self.DiGraph.nodes[name]["input"] = {}

    def add_fixed_node(self, name: str, input:Any) -> None:
        """
        Method to add a fixed node to the DAG. Fixed nodes are design matrices.

        Args:
            name (str): Name of the fixed node.
            input (Any): Jax.numpy array.
        """

        attr = {"value": input.design_matrix}
        self.DiGraph.add_node(name, attr=attr)
        self.DiGraph.nodes[name]["node_type"] = "fixed"

    def add_hyper_node(self, name:str, input: Any) -> None:
        """
        Method to add a hyperparameter node.

        Args:
            name (str): Name of the hyperparameter node.
            input (Any): Jax.numpy array.
        """

        attr = {"value": input.value}
        self.DiGraph.add_node(name, attr=attr)
        self.DiGraph.nodes[name]["node_type"] = "hyper"

    def update_lpred(self, node: str, attr: dict) -> Array:
        """
        Method to calculate the value of linear predictor node.

        Args:
            node (str): Name of the linear predictor node.
            attr (dict): Attributes of the node.

        Returns:
            Array: Values for the linear predictor after applying the bijector.
        """

        bijector = attr["bijector"]
        design_matrix = self.DiGraph.nodes[node]["input"]["fixed"]
        input = self.DiGraph.nodes[node]["input"]
        values_params = jnp.concatenate([input for key, input in input.items() if key != "fixed"])

        nu = dot(design_matrix, values_params)

        if bijector is not None:
            transformed = bijector(nu)
        else:
            transformed = nu

        return transformed

    def update_param_value(self, attr: dict, sample: Array, transform: bool = True) -> Array:
        """
        Method to calculate the value attribute of a parameter node using samples from a variational distribution for a parameter in the DAG.

        Args:
            attr (dict): Attributes of a node
            sample (Array): Sample (internal representation) from the variational distribution of the parameter.
            transform (Any): A bijector function that respects the parameter space of the paramter.

        Returns:
            Array: Sample of the parameter after applying the bijector.
        """

        bijector = attr["bijector"]

        if bijector is not None and transform == True:
            transformed = bijector(sample)
        else:
            transformed = sample

        return transformed

    def init_dist(self, dist: Distribution, input: dict) -> Distribution:
        """
        Method to initialize the probability distribution of a strong node (one with a probability distribution).

        Args:
            dist (Distribution): A tensorflow probability distribution.
            input (dict): Key, value pair, where keys should match the names of the parameters of
            the distribution.

        Returns:
            Distribution: A initialized tensorflow probability distribution.
        """

        return dist(**input)

    def init_traversal(self) -> None:
        """
        Method to initialize the traversal order for the topological sort algorithm.
        """

        self.traversal_order = list(nx.topological_sort(self.DiGraph))

    def init_update_traversal(self) -> None:
        """
        Method to filter out all probabilistic nodes and linear predictor nodes from the traversal order.
        """

        for node in self.traversal_order:
            node_type = self.DiGraph.nodes[node].get("node_type")
            if node_type in ["strong", "lpred", "root"]:
                self.update_traversal_order.append(node)

    def init_prob_traversal(self) -> None:
        """
        Method to filter out all probabilistic nodes from the traversal order.
        """

        for node in self.traversal_order:
            node_type = self.DiGraph.nodes[node].get("node_type")
            if node_type in ["strong", "root"]:
                self.prob_traversal_order.append(node)

    def logprior(self, dist: Distribution, sample: Array) -> Array:
        """
        Method to calculate the log-prior probability of a parameter node.

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            sample (Array): Samples from the variational distribution for the parameter.

        Returns:
            Array: Log-prior probabilities of the parameters.
        """

        return dist.log_prob(sample)

    def loglik(self, dist: Distribution, value: Array) -> Array:
        """
        Method to calculate the log-likelihood of the response (root node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (Array): Values of the response.

        Returns:
            Array: Log-likelihood of the response.
        """

        return dist.log_prob(value)

    def logprob(self) -> Array:

        log_prob_sum = jnp.array([0.0], dtype=jnp.float32)

        for node in self.prob_traversal_order:
            node_type = self.DiGraph.nodes[node].get("node_type")
            if node_type == "strong":
                log_prior = self.DiGraph.nodes[node].get("attr", {}).get("log_prior", 0.0)
                log_prob_sum += jnp.sum(log_prior)
            elif node_type == "root":
                log_lik = self.DiGraph.nodes[node].get("attr", {}).get("log_lik", 0.0)
                log_prob_sum += jnp.sum(log_lik)

        return log_prob_sum

    def init_graph(self, sample: dict) -> None:
        """
        Method to update the full DAG.

        Args:
            sample (dict): Samples from the variational distribution for the params in the model.
        """

        # Use topological sort algorithm
        for node in self.traversal_order:
            node_type = self.DiGraph.nodes[node]["node_type"]
            attr = self.DiGraph.nodes[node]["attr"]
            successors = self.DiGraph.successors(node)

            if node_type == "hyper":
                input = attr["value"]

                for successor in successors:
                    edge = self.DiGraph.get_edge_data(node, successor)
                    self.DiGraph.nodes[successor]["input"][edge["role"]] = input
            elif node_type == "fixed":
                input = attr["value"]

                for successor in successors:
                    edge = self.DiGraph.get_edge_data(node, successor)
                    self.DiGraph.nodes[successor]["input"][edge["role"]] = input
            elif node_type == "lpred":
                attr["value"] = self.update_lpred(node, attr)
                input = attr["value"]

                for successor in successors:
                    self.DiGraph.nodes[successor]["input"][node] = input
            elif node_type == "strong":
                attr["dist"] = self.init_dist(attr["dist"], self.DiGraph.nodes[node]["input"])
                attr["value"] = self.update_param_value(attr, sample[node], transform=False)
                attr["log_prior"] = self.logprior(attr["dist"], attr["value"])
                input = attr["value"]

                for successor in successors:
                    self.DiGraph.nodes[successor]["input"][node] = input
            elif node_type == "root":
                init_dist = self.init_dist(attr["dist"], self.DiGraph.nodes[node]["input"])
                attr["log_lik"] = self.loglik(init_dist, attr["value"])

    def build_graph(self) -> None:
        """
        Method to build the DAG from class Model.
        """

        # Store current values of all parameters to init graph
        params = {}

        # Initialize the root of the graph
        self.add_root()

        # Build graph by searching through the classes
        for kw1, input1 in self.Model.response_dist.kwinputs.items():
            if isinstance(input1, Lpred):
                self.add_lpred_node(name=kw1, input=input1)
                self.DiGraph.add_edge(kw1, "response", role="lpred", bijector=input1.function)
                name = input1.Obs.name
                self.add_fixed_node(name=name, input=input1)
                self.DiGraph.add_edge(name, kw1, role="fixed")

                for kw2, input2 in input1.kwinputs.items():
                    self.add_strong_node(name=kw2, input=input2)
                    self.DiGraph.add_edge(kw2, kw1, role="param")

                    for kw3, input3 in input2.distribution.kwinputs.items():
                        name = input3.name
                        self.add_hyper_node(name=name, input=input3)
                        self.DiGraph.add_edge(name, kw2, role=kw3)

                    params[kw2] = input2.value

            elif isinstance(input1, Param):
                self.add_strong_node(name=kw1, input=input1)
                self.DiGraph.add_edge(kw1, "response", role="param")

                for kw3, input3 in input1.distribution.kwinputs.items():
                    name = input3.name
                    self.add_hyper_node(name=name, input=input3)
                    self.DiGraph.add_edge(name, kw1, role=kw3)

                params[kw1] = input1.value

        # Obtain the traversal order for the initialization of the graph
        self.init_traversal()

        # Obtain the traversal order to update the Graph i.e. for all probabilistic nodes and linear predictor nodes
        self.init_update_traversal()

        # Obtain the traversal order for all probabilistic nodes
        self.init_prob_traversal()

        # Initialize the full graph once
        self.init_graph(params)

    def update_graph(self, sample: dict) -> None:

        for node in self.update_traversal_order:
            node_type = self.DiGraph.nodes[node]["node_type"]
            attr = self.DiGraph.nodes[node]["attr"]
            successors = self.DiGraph.successors(node)

            if node_type == "lpred":
                attr["value"] = self.update_lpred(node, attr)
                input = attr["value"]

                for successor in successors:
                    self.DiGraph.nodes[successor]["input"][node] = input
            elif node_type == "strong":
                attr["value"] = self.update_param_value(attr, sample[node], transform=True)
                attr["log_prior"] = self.logprior(attr["dist"], attr["value"])
                input = attr["value"]

                for successor in successors:
                    self.DiGraph.nodes[successor]["input"][node] = input
            elif node_type == "root":
                init_dist = self.init_dist(attr["dist"], self.DiGraph.nodes[node]["input"])
                attr["log_lik"] = self.loglik(init_dist, attr["value"])

    def visualize_graph(self) -> None:
        """
        Method to visualize the DAG.
        """

        pos = nx.nx_agraph.graphviz_layout(self.DiGraph, prog="dot")
        nx.set_node_attributes(self.DiGraph, pos, "pos")

        for node in self.DiGraph.nodes:
            self.DiGraph.nodes[node]["label"] = f"{node}"

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Model Graph")
        ax.set_axis_off()

        pos = nx.get_node_attributes(self.DiGraph, "pos")
        labels = nx.get_node_attributes(self.DiGraph, "label")
        node_type = nx.get_node_attributes(self.DiGraph, "node_type")

        node_shapes = {
            "strong": "o",
            "weak": ""
        }

        node_labels = {node: label for node, label in labels.items() if node in node_type}

        strong_nodes = [node for node, node_type in node_type.items() if node_type == "strong" or node_type == "root"]
        weak_nodes = [node for node, node_type in node_type.items() if node_type != "strong"]

        nx.draw_networkx_edges(self.DiGraph, pos, ax=ax, node_size=1000)
        nx.draw_networkx_labels(self.DiGraph, pos, ax=ax, labels=node_labels, font_size=10)

        nx.draw_networkx_nodes(self.DiGraph, pos, nodelist=strong_nodes, ax=ax, node_color="lightblue",
                               node_shape=node_shapes["strong"], node_size=1000)
        nx.draw_networkx_nodes(self.DiGraph, pos, nodelist=weak_nodes, ax=ax, node_color="lightblue",
                               node_shape=node_shapes["weak"], node_size=1000)

        edge_labels = nx.get_edge_attributes(self.DiGraph, "bijector")

        for key, value in edge_labels.items():
            if value is not None:
                string_to_search = str(edge_labels[key])
                pattern = r"jax\.numpy\.(\w+)"
                match_pattern = re.search(pattern, string_to_search)
                edge_labels[key] = match_pattern.group()
            else:
                edge_labels[key] = ""

        nx.draw_networkx_edge_labels(self.DiGraph, pos, edge_labels=edge_labels, ax=ax, font_size=8)

        plt.show()
