"""
Directed Graph.
"""

import jax
import jax.numpy as jnp

import networkx as nx
import matplotlib.pyplot as plt
import re

from typing import Any, Union

from .model import (
    Model,
    Lpred,
    Hyper,
    Param,
    Dist,
    Obs,
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

        attr = {"dim": self.Model.response.shape[0],
                "value": self.Model.response,
                "dist": self.Model.response_dist.get_dist(),
                "log_lik": self.Model.log_lik}
        self.DiGraph.add_node("response", attr=attr)
        self.DiGraph.nodes["response"]["node_type"] = "root"
        self.DiGraph.nodes["response"]["input"] = {}

    def add_strong_node(self, name: str, param: Param) -> None:
        """
        Method to add a strong node to the DAG. Strong nodes have a probability distribution.

        Args:
            name (str): Name of the node. Key of kwinputs.
            param (Param): Class Param.
        """

        attr = {"dim": param.dim,
                "param_space": param.param_space,
                "value": param.value,
                "dist": param.distribution.get_dist(),
                "fixed_params": param.distribution.fixed_params,
                "log_prior": param.log_prior}
        self.DiGraph.add_node(name, attr=attr)
        self.DiGraph.nodes[name]["node_type"] = "strong"
        self.DiGraph.nodes[name]["input"] = {}
        self.DiGraph.nodes[name]["input_fixed"] = True

    def add_lpred_node(self, name: str, lpred: Lpred) -> None:
        """
        Method to add a linear predictor (lpred) node to the DAG. Lpred nodes are linear predictors
        with a bijector function (inverse link function).

        Args:
            name (str): Name of the weak node.
            lpred (Lpred): Class Lpred.
        """

        attr = {"value": lpred.value,
                 "bijector": lpred.function}
        self.DiGraph.add_node(name, attr=attr)
        self.DiGraph.nodes[name]["node_type"] = "lpred"
        self.DiGraph.nodes[name]["input"] = {}

    def add_fixed_node(self, name: str, obs: Obs) -> None:
        """
        Method to add a fixed node to the DAG. Fixed nodes are design matrices.

        Args:
            name (str): Name of the fixed node.
            obs (Obs): Class Obs.
        """

        attr = {"value": obs.design_matrix}
        self.DiGraph.add_node(name, attr=attr)
        self.DiGraph.nodes[name]["node_type"] = "fixed"

    def add_hyper_node(self, name: str, hyper: Hyper) -> None:
        """
        Method to add a hyperparameter node.

        Args:
            name (str): Name of the hyperparameter node.
            hyper (Hyper): Class Hyper.
        """

        attr = {"value": hyper.value}
        self.DiGraph.add_node(name, attr=attr)
        self.DiGraph.nodes[name]["node_type"] = "hyper"

    def init_dist(self, dist: Distribution,
                  params: dict,
                  fixed_params: dict | None = None) -> Distribution:
        """
        Method to initialize the probability distribution of a strong node (one with a probability distribution).

        Args:
            dist (Distribution): A tensorflow probability distribution.
            params (dict): Key, value pair, where keys should match the names of the parameters of
            the distribution.

        Returns:
            Distribution: A initialized tensorflow probability distribution.
        """

        if fixed_params is None:
            return dist(**params)
        else:
            return dist(**params, **fixed_params)


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

    def update_lpred(self, node: str, attr: dict, idx: Union[Array, None] = None) -> Array:
        """
        Method to calculate the value of linear predictor node.

        Args:
            node (str): Name of the linear predictor node.
            attr (dict): Attributes of the node.

        Returns:
            Array: Values for the linear predictor after applying the bijector.
        """

        bijector = attr["bijector"]
        if idx is not None:
            design_matrix = self.DiGraph.nodes[node]["input"]["fixed"][idx,]
        else:
            design_matrix = self.DiGraph.nodes[node]["input"]["fixed"]
        design_matrix = jnp.expand_dims(design_matrix, 0)

        input_pass = self.DiGraph.nodes[node]["input"]

        values_params = jnp.concatenate([item for key, item in input_pass.items() if key != "fixed"], axis=-1)
        values_params = jnp.expand_dims(values_params, -1)

        nu = jnp.matmul(design_matrix, values_params)
        nu = jnp.squeeze(nu, axis=-1)

        if bijector is not None:
            transformed = bijector(nu)
        else:
            transformed = nu

        return transformed

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

    def loglik(self, dist: Distribution, value: Array, idx: Union[Array, None] = None) -> Array:
        """
        Method to calculate the log-likelihood of the response (root node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (Array): Values of the response.

        Returns:
            Array: Log-likelihood of the response.
        """

        if idx is not None:
            log_lik = dist.log_prob(value[idx])
        else:
            log_lik = dist.log_prob(value)

        return log_lik

    def logprob(self) -> Array:
        """
        Method to calculate the log-probability of the graph. Hence summing the log-likelihood
        and all log-priors of probabilistic nodes.
        """

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
            successors = list(self.DiGraph.successors(node))

            # Question there could also be several sucessors for a node?
            if successors:
                successor = successors[0]

            if node_type == "hyper":
                input_pass = attr["value"]

                edge = self.DiGraph.get_edge_data(node, successor)
                self.DiGraph.nodes[successor]["input"][edge["role"]] = input_pass

            elif node_type == "fixed":
                input_pass = attr["value"]

                edge = self.DiGraph.get_edge_data(node, successor)
                self.DiGraph.nodes[successor]["input"][edge["role"]] = input_pass

            elif node_type == "lpred":
                attr["value"] = self.update_lpred(node, attr)
                input_pass = attr["value"]

                edge = self.DiGraph.get_edge_data(node, successor)
                self.DiGraph.nodes[successor]["input"][edge["role"]] = input_pass

            elif node_type == "strong":
                attr["value"] = sample[node]

                for predecessor in self.DiGraph.predecessors(node):
                    if self.DiGraph.nodes[predecessor]["node_type"] != "hyper":
                        self.DiGraph.nodes[node]["input_fixed"] = False
                        break

                if self.DiGraph.nodes[node]["input_fixed"]:
                    attr["dist"] = self.init_dist(dist=attr["dist"],
                                                  params=self.DiGraph.nodes[node]["input"],
                                                  fixed_params=attr["fixed_params"])
                else:
                    init_dist = self.init_dist(dist=attr["dist"],
                                               params=self.DiGraph.nodes[node]["input"],
                                               fixed_params=attr["fixed_params"])
                    attr["log_prior"] = self.logprior(init_dist, attr["value"])

                input_pass = attr["value"]
                edge = self.DiGraph.get_edge_data(node, successor)
                self.DiGraph.nodes[successor]["input"][edge["role"]] = input_pass

            elif node_type == "root":
                init_dist = self.init_dist(attr["dist"], self.DiGraph.nodes[node]["input"])
                attr["log_lik"] = self.loglik(init_dist, attr["value"])

    def build_graph_recursive(self, obj: Any, parent_node: str, node_name: str,
                              params: dict) -> None:

        if isinstance(obj, Lpred):
            # Lpred case
            self.add_lpred_node(name=node_name, lpred=obj)
            self.DiGraph.add_edge(node_name, parent_node, role=node_name,
                                  bijector=obj.function)
            name_fixed = obj.Obs.name
            self.add_fixed_node(name=name_fixed, obs=obj)
            self.DiGraph.add_edge(name_fixed, node_name, role="fixed")

        elif isinstance(obj, Param):
            # Param case
            name_param = obj.name
            self.add_strong_node(name=name_param, param=obj)
            self.DiGraph.add_edge(name_param, parent_node, role=node_name)
            node_name = name_param
            params[name_param] = obj.value

        elif isinstance(obj, Hyper):
            # Hyper case
            name_hyper = obj.name
            self.add_hyper_node(name=name_hyper, hyper=obj)
            self.DiGraph.add_edge(name_hyper, parent_node, role=node_name)

        if isinstance(obj, (list, tuple)):
            for item in obj:
                self.build_graph_recursive(obj=item,
                                           parent_node=parent_node,
                                           node_name=node_name,
                                           params=params)

        if isinstance(obj, dict):
            for key, value in obj.items():
                self.build_graph_recursive(obj=value,
                                           parent_node=node_name,
                                           node_name=key,
                                           params=params)

        elif hasattr(obj, "__dict__"):
            for value in obj.__dict__.values():
                self.build_graph_recursive(obj=value,
                                           parent_node=parent_node,
                                           node_name=node_name,
                                           params=params)

    def build_graph(self) -> None:
        """
        Method to build the DAG from class Model.
        """

        # Store current values of all parameters to init graph
        params = {}

        # Initialize the root of the graph
        self.add_root()

        # if isinstance(obj, Lpred):

        # Build graph by searching recursively through the classes
        self.build_graph_recursive(obj=self.Model.response_dist,
                                   parent_node="response",
                                   node_name="response",
                                   params=params)

        # Obtain the traversal order for the initialization of the graph
        self.init_traversal()

        # Obtain the traversal order to update the Graph i.e. for all probabilistic nodes and linear predictor nodes
        self.init_update_traversal()

        # Obtain the traversal order for all probabilistic nodes
        self.init_prob_traversal()

        # Initialize the full graph once
        self.init_graph(params)

        self.params = params

    def update_graph(self, samples: dict, idx: Union[Array, None] = None) -> None:

        for node in self.update_traversal_order:
            node_type = self.DiGraph.nodes[node]["node_type"]
            attr = self.DiGraph.nodes[node]["attr"]
            successors = list(self.DiGraph.successors(node))

            # Question there could also be several sucessors for a node?
            if successors:
                successor = successors[0]

            if node_type == "lpred":
                attr["value"] = self.update_lpred(node, attr, idx)

                edge = self.DiGraph.get_edge_data(node, successor)
                self.DiGraph.nodes[successor]["input"][edge["role"]] = attr["value"]

            elif node_type == "strong":
                attr["value"] = samples[node]
                if self.DiGraph.nodes[node]["input_fixed"]:
                    attr["log_prior"] = self.logprior(attr["dist"], attr["value"])
                else:
                    init_dist = self.init_dist(dist=attr["dist"],
                                               params=self.DiGraph.nodes[node]["input"],
                                               fixed_params=attr["fixed_params"])
                    attr["log_prior"] = self.logprior(init_dist, attr["value"])

                # Test the dims
                # print(node + ":", attr["log_prior"].shape)
                edge = self.DiGraph.get_edge_data(node, successor)
                self.DiGraph.nodes[successor]["input"][edge["role"]] = attr["value"]

            elif node_type == "root":
                init_dist = self.init_dist(dist=attr["dist"],
                                           params=self.DiGraph.nodes[node]["input"])
                attr["log_lik"] = self.loglik(init_dist, attr["value"], idx)

                # Test the dims
                # print("response" + ":", attr["log_lik"].shape)

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
