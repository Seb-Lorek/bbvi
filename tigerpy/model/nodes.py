"""
Directed Graph.
"""

import networkx as nx

import jax.numpy as jnp
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
    def __init__(self, model: Model):
        self.digraph = nx.DiGraph()
        self.model = model
        self.traversal_order = None
        self.update_traversal_order = []
        self.prob_traversal_order = []

    def add_root(self) -> None:
        """
        Method to generate the root of the DAG (response).
        """

        attr = {"value": self.model.response,
                "dim": self.model.response.shape[0],
                "dist": self.model.response_dist.get_dist(),
                "log_lik": self.model.log_lik}
        self.digraph.add_node("response", attr=attr)
        self.digraph.nodes["response"]["node_type"] = "root"
        self.digraph.nodes["response"]["input"] = {}

    def add_strong_node(self, name: str, param: Param) -> None:
        """
        Method to add a strong node to the DAG. Strong nodes have a probability distribution.

        Args:
            name (str): Name of the node. Key of kwinputs.
            param (Param): Initialization of class Param.
        """

        attr = {"value": param.value,
                "dim": param.dim,
                "param_space": param.param_space,
                "dist": param.distribution.get_dist(),
                "additional_params": param.distribution.additional_params,
                "log_prior": param.log_prior}
        self.digraph.add_node(name, attr=attr)
        self.digraph.nodes[name]["node_type"] = "strong"
        self.digraph.nodes[name]["input"] = {}
        self.digraph.nodes[name]["input_fixed"] = True

    def add_lpred_node(self, name: str, lpred: Lpred) -> None:
        """
        Method to add a linear predictor (lpred) node to the DAG. Lpred nodes are linear predictors
        with a bijector function (inverse link function).

        Args:
            name (str): Name of the linear predictor node.
            lpred (Lpred): Initialization of class Lpred.
        """

        attr = {"value": lpred.value,
                 "bijector": lpred.function}
        self.digraph.add_node(name, attr=attr)
        self.digraph.nodes[name]["node_type"] = "lpred"
        self.digraph.nodes[name]["input"] = {}

    def add_fixed_node(self, name: str, obs: Obs) -> None:
        """
        Method to add a fixed node to the DAG. Fixed nodes contain design matrices.

        Args:
            name (str): Name of the fixed node.
            obs (Obs): Initialization of class Obs.
        """

        attr = {"value": jnp.asarray(obs.design_matrix, dtype=jnp.float32)}
        self.digraph.add_node(name, attr=attr)
        self.digraph.nodes[name]["node_type"] = "fixed"

    def add_hyper_node(self, name: str, hyper: Hyper) -> None:
        """
        Method to add a hyperparameter node.

        Args:
            name (str): Name of the hyperparameter node.
            hyper (Hyper): Initialization of class Hyper.
        """

        attr = {"value": hyper.value}
        self.digraph.add_node(name, attr=attr)
        self.digraph.nodes[name]["node_type"] = "hyper"

    def init_dist(self,
                  dist: Distribution,
                  params: dict,
                  additional_params: Union[dict, None] = None) -> Distribution:
        """
        Method to initialize the probability distribution of a strong node (node with a probability distribution).

        Args:
            dist (Distribution): A tensorflow probability distribution.
            params (dict): Key, value pair, where keys should match the names of the parameters of
            the distribution.
            additional_params (dict): Additional parameters of a distribution, currently
            implemented to store the penalty matrices for the MultivariateNormalDegenerate
            distribution.

        Returns:
            Distribution: A initialized tensorflow probability distribution.
        """

        if additional_params is None:
            initialized_dist = dist(**params)
        else:
            initialized_dist = dist(**params, **additional_params)

        return initialized_dist

    def init_traversal(self) -> None:
        """
        Method to initialize the traversal order for the topological sort algorithm.
        """

        self.traversal_order = list(nx.topological_sort(self.digraph))

    def init_update_traversal(self) -> None:
        """
        Method to filter out all probabilistic nodes and linear predictor nodes from the traversal order.
        """

        for node in self.traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            if node_type in ["strong", "lpred", "root"]:
                self.update_traversal_order.append(node)

    def init_prob_traversal(self) -> None:
        """
        Method to filter out all probabilistic nodes from the traversal order.
        """

        for node in self.traversal_order:
            node_type = self.digraph.nodes[node].get("node_type")
            if node_type in ["strong", "root"]:
                self.prob_traversal_order.append(node)

    def update_lpred(self, node: str, attr: dict, batch_idx: Union[Array, None] = None) -> Array:
        """
        Method to calculate the value of a linear predictor node.

        Args:
            node (str):  Name of the linear predictor node.
            attr (dict): Attributes of the node.
            batch_idx (Union[Array, None], optional): Array of indices to select a
            subset of the observations. If None all observations are used. Defaults to None.

        Returns:
            Array: Values for the linear predictor after applying the bijector (inverse link function).
        """

        bijector = attr["bijector"]

        if batch_idx is not None:
            design_matrix = self.digraph.nodes[node]["input"]["fixed"][batch_idx,:]
        else:
            design_matrix = self.digraph.nodes[node]["input"]["fixed"]
        # print("design_matrix before:", design_matrix.shape)

        design_matrix = jnp.expand_dims(design_matrix, axis=0)
        # print("design_matrix after:", design_matrix.shape)

        input_pass = self.digraph.nodes[node]["input"]

        values_params = jnp.concatenate([item for key, item in input_pass.items() if key != "fixed"], axis=-1)
        values_params = jnp.expand_dims(values_params, -1)
        # print("Reg coef before:", values_params.shape)

        nu = jnp.matmul(design_matrix, values_params)
        # print("nu before:", nu.shape)

        nu = jnp.squeeze(nu)
        # print("nu after:", nu.shape)

        if bijector is not None:
            transformed = bijector(nu)
        else:
            transformed = nu

        return transformed

    def logprior(self, dist: Distribution, value: Array) -> Array:
        """
        Method to calculate the log-prior probability of a parameter node (strong node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (Array): Value of the parameter.

        Returns:
            Array: Log-prior probabilities of the parameters.
        """

        return dist.log_prob(value)

    def loglik(self, dist: Distribution, value: Array, batch_idx: Union[Array, None] = None) -> Array:
        """
        Method to calculate the log-likelihood of the response (root node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (Array): Values of the response.
            batch_idx (Union[Array, None], optional): Array of indices to select a
            subset of the observations. If None all observations are used. Defaults to None.

        Returns:
            Array: Log-likelihood of the response.
        """

        if batch_idx is not None:
            log_lik = dist.log_prob(value[batch_idx])
        else:
            log_lik = dist.log_prob(value)

        # print("log_lik:", log_lik.shape)
        return log_lik

    # new function to collect all prior probabilities in the graph
    def collect_logpriors(self, num_samples: int = 1) -> Array:

        log_prior_sum = jnp.zeros((num_samples,), dtype=jnp.float32)

        for node in self.prob_traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]

            if node_type == "strong":
                log_prior = self.digraph.nodes[node]["attr"].get("log_prior", 0.0)

                # Check if really shape[1] or shape[-1]
                if log_prior.ndim == 2 and log_prior.shape[1] == 1:
                    log_prior = jnp.squeeze(log_prior)
                elif log_prior.ndim == 2 and log_prior.shape[1] != 1:
                    log_prior = jnp.sum(log_prior, axis=-1)

                log_prior_sum += log_prior

        return log_prior_sum

    def build_graph_recursive(self,
                              obj: Any,
                              parent_node: str,
                              node_name: str,
                              params: dict) -> None:
        """
        Method to build the DAG recursively. Calls methods add_* to create the
        different node types.

        Args:
            obj (Any): Classes of `model.py` i.e. Hyper, Param, Lpred.
            parent_node (str): Name of the parent node.
            node_name (str): Name of the current node.
            params (dict): Dictionary that gets filled with the initial supplied
            values.
        """

        if isinstance(obj, Lpred):
            # Lpred case
            self.add_lpred_node(name=node_name, lpred=obj)
            self.digraph.add_edge(node_name, parent_node, role=node_name,
                                  bijector=obj.function)
            name_fixed = obj.obs.name
            self.add_fixed_node(name=name_fixed, obs=obj.obs)
            self.digraph.add_edge(name_fixed, node_name, role="fixed")

        elif isinstance(obj, Param):
            # Param case
            name_param = obj.name
            self.add_strong_node(name=name_param, param=obj)
            self.digraph.add_edge(name_param, parent_node, role=node_name)
            node_name = name_param
            params[name_param] = obj.value

        elif isinstance(obj, Hyper):
            # Hyper case
            name_hyper = obj.name
            self.add_hyper_node(name=name_hyper, hyper=obj)
            self.digraph.add_edge(name_hyper, parent_node, role=node_name)

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

    def init_graph(self, sample: dict) -> None:
        """
        Method to initialize the DAG.

        Args:
            sample (dict): Samples from the variational distribution for
            the params in the model.
        """

        # Use the topological sort algorithm
        for node in self.traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            attr = self.digraph.nodes[node]["attr"]
            successors = list(self.digraph.successors(node))

            # ToDo: There could also be several successors for a node?
            # F.e. a parameter that couples two parameters ?
            if successors:
                # Select first successor and neglect all other
                successor = successors[0]

            if node_type == "hyper":
                input_pass = attr["value"]

                edge = self.digraph.get_edge_data(node, successor)
                self.digraph.nodes[successor]["input"][edge["role"]] = input_pass

            elif node_type == "fixed":
                input_pass = attr["value"]

                edge = self.digraph.get_edge_data(node, successor)
                self.digraph.nodes[successor]["input"][edge["role"]] = input_pass

            elif node_type == "lpred":
                attr["value"] = self.update_lpred(node, attr)
                input_pass = attr["value"]

                edge = self.digraph.get_edge_data(node, successor)
                self.digraph.nodes[successor]["input"][edge["role"]] = input_pass

            elif node_type == "strong":
                attr["value"] = sample[node]

                for predecessor in self.digraph.predecessors(node):
                    if self.digraph.nodes[predecessor]["node_type"] not in ["hyper", "fixed"]:
                        self.digraph.nodes[node]["input_fixed"] = False
                        break

                # If all inputs are fixed we need to initialize the probabiltiy
                # distribution only onces.
                if self.digraph.nodes[node]["input_fixed"]:
                    attr["dist"] = self.init_dist(dist=attr["dist"],
                                                  params=self.digraph.nodes[node]["input"],
                                                  additional_params=attr["additional_params"])
                else:
                    init_dist = self.init_dist(dist=attr["dist"],
                                               params=self.digraph.nodes[node]["input"],
                                               additional_params=attr["additional_params"])
                    attr["log_prior"] = self.logprior(init_dist, attr["value"])

                input_pass = attr["value"]
                edge = self.digraph.get_edge_data(node, successor)
                self.digraph.nodes[successor]["input"][edge["role"]] = input_pass

            elif node_type == "root":
                init_dist = self.init_dist(attr["dist"], self.digraph.nodes[node]["input"])
                attr["log_lik"] = self.loglik(init_dist, attr["value"])

    def build_graph(self) -> None:
        """
        Method to build the DAG from class Model.
        """

        # Store current values of all parameters to init graph
        params = {}

        # Initialize the root of the graph
        self.add_root()

        # Build graph by searching recursively through the classes
        self.build_graph_recursive(obj=self.model.response_dist,
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

    # check how to improve this function
    def update_graph(self, samples: dict, batch_idx: Union[Array, None] = None) -> None:
        """
        Method to update the graph with samples from the variational distribution.

        Args:
            samples (dict): Dictionary containing the samples from the variational
            distribution.
            batch_idx (Union[Array, None], optional): Array of indices to select a
            subset of the observations. If None all observations are used. Defaults to None.
        """

        for node in self.update_traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            attr = self.digraph.nodes[node]["attr"]
            successors = list(self.digraph.successors(node))

            # Question there could also be several sucessors for a node?
            if successors:
                successor = successors[0]

            if node_type == "lpred":
                attr["value"] = self.update_lpred(node, attr, batch_idx)

                edge = self.digraph.get_edge_data(node, successor)
                self.digraph.nodes[successor]["input"][edge["role"]] = attr["value"]

            elif node_type == "strong":
                attr["value"] = samples[node]
                if self.digraph.nodes[node]["input_fixed"]:
                    attr["log_prior"] = self.logprior(attr["dist"], attr["value"])
                else:
                    init_dist = self.init_dist(dist=attr["dist"],
                                               params=self.digraph.nodes[node]["input"],
                                               additional_params=attr["additional_params"])
                    attr["log_prior"] = self.logprior(init_dist, attr["value"])

                edge = self.digraph.get_edge_data(node, successor)
                self.digraph.nodes[successor]["input"][edge["role"]] = attr["value"]

            elif node_type == "root":
                init_dist = self.init_dist(dist=attr["dist"],
                                           params=self.digraph.nodes[node]["input"])
                attr["log_lik"] = self.loglik(init_dist, attr["value"], batch_idx)

    def visualize_graph(self) -> None:
        """
        Method to visualize the DAG.
        """

        pos = nx.nx_agraph.graphviz_layout(self.digraph, prog="dot")
        nx.set_node_attributes(self.digraph, pos, "pos")

        for node in self.digraph.nodes:
            self.digraph.nodes[node]["label"] = f"{node}"

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Model Graph")
        ax.set_axis_off()

        pos = nx.get_node_attributes(self.digraph, "pos")
        labels = nx.get_node_attributes(self.digraph, "label")
        node_type = nx.get_node_attributes(self.digraph, "node_type")

        node_shapes = {
            "strong": "o",
            "weak": ""
        }

        node_labels = {node: label for node, label in labels.items() if node in node_type}

        strong_nodes = [node for node, node_type in node_type.items() if node_type == "strong" or node_type == "root"]
        weak_nodes = [node for node, node_type in node_type.items() if node_type != "strong"]

        nx.draw_networkx_edges(self.digraph, pos, ax=ax, node_size=1000)
        nx.draw_networkx_labels(self.digraph, pos, ax=ax, labels=node_labels, font_size=10)

        nx.draw_networkx_nodes(self.digraph, pos, nodelist=strong_nodes, ax=ax, node_color="lightblue",
                               node_shape=node_shapes["strong"], node_size=1000)
        nx.draw_networkx_nodes(self.digraph, pos, nodelist=weak_nodes, ax=ax, node_color="lightblue",
                               node_shape=node_shapes["weak"], node_size=1000)

        edge_labels = nx.get_edge_attributes(self.digraph, "bijector")

        for key, value in edge_labels.items():
            if value is not None:
                string_to_search = str(edge_labels[key])
                pattern = r"jax\.numpy\.(\w+)"
                match_pattern = re.search(pattern, string_to_search)
                edge_labels[key] = match_pattern.group()
            else:
                edge_labels[key] = ""

        nx.draw_networkx_edge_labels(self.digraph, pos, edge_labels=edge_labels, ax=ax, font_size=8)

        plt.show()
