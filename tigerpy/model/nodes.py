"""
Directed graph.
"""

import jax.numpy as jnp
import jax 

import networkx as nx

import matplotlib.pyplot as plt

import re
import copy
from typing import (
    Any,
    Union
)

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
    """
    Class to build a DAG.
    """

    def __init__(self, 
                 model: Model):
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
                "dim": self.model.num_obs,
                "dist": self.model.response_dist.get_dist(),
                "log_lik": self.model.log_lik}
        self.digraph.add_node("response", attr=attr)
        self.digraph.nodes["response"]["node_type"] = "root"
        self.digraph.nodes["response"]["input"] = {}

    def add_strong_node(self, 
                        name: str, 
                        param: Param) -> None:
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

    def add_fixed_node(self, 
                       name: str, 
                       obs: Obs) -> None:
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
                  params: dict) -> Distribution:
        """
        Method to initialize the probability distribution of a strong node (node with a probability distribution).

        Args:
            dist (Distribution): A tensorflow probability distribution.
            params (dict): Key, value pair, where keys should match the names of the parameters of
            the distribution.

        Returns:
            Distribution: A initialized tensorflow probability distribution.
        """

        initialized_dist = dist(**params)

        return initialized_dist

    def init_traversal(self) -> None:
        """
        Method to initialize the traversal order for the topological sort algorithm.
        """

        self.traversal_order = list(nx.topological_sort(self.digraph))

    def init_update_traversal(self) -> None:
        """
        Method to filter out all probabilistic and linear predictor nodes from the traversal order.
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
            node_type = self.digraph.nodes[node]["node_type"]
            if node_type in ["strong", "root"]:
                self.prob_traversal_order.append(node)

    def update_lpred(self, 
                     node: str, 
                     attr: dict) -> Array:
        """
        Method to calculate the value of a linear predictor node.

        Args:
            node (str):  Name of the linear predictor node.
            attr (dict): Attributes of the node.

        Returns:
            Array: Values for the linear predictor after applying the bijector (inverse link function).
        """

        bijector = attr["bijector"]
        design_matrix = self.digraph.nodes[node]["input"]["fixed"]
        input_pass = self.digraph.nodes[node]["input"]
        values_params = jnp.concatenate([item for key, item in input_pass.items() if key != "fixed"], axis=-1)
        nu = jnp.dot(design_matrix, values_params)

        if bijector is not None:
            transformed = bijector(nu)
        else:
            transformed = nu

        return transformed

    def logprior(self, 
                 dist: Distribution, 
                 value: jax.Array) -> jax.Array:
        """
        Method to calculate the log-prior probability of a parameter node (strong node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (jax.Array): Value of the parameter.

        Returns:
            jax.Array: Prior log-probabilities of the parameters.
        """

        return dist.log_prob(value)

    def loglik(self,
               dist: Distribution,
               value: jax.Array) -> jax.Array:
        """
        Method to calculate the log-likelihood of the response (root node).

        Args:
            dist (Distribution): A initialized tensorflow probability distribution.
            value (jax.Array): Values of the response.

        Returns:
            jax.Array: Log-likelihood of the response.
        """

        return dist.log_prob(value)

    def sum_logpriors(self) -> jax.Array:
        """
        Method to sum all the prior log-probabilities of the DAG, i.e. collect them 
        from the strong nodes.

        Returns:
            jax.Array: The sum of all prior log-probabilities.
        """

        log_prior_sum = jnp.array(0.0, dtype=jnp.float32)
        
        for node in self.prob_traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]

            if node_type == "strong":
                log_prior = self.digraph.nodes[node]["attr"].get("log_prior", 0.0)
                log_prior_sum += jnp.sum(log_prior)

        return log_prior_sum

    def logprob(self) -> Array:
        """
        Method to calculate the log-probability of the DAG.
        """

        log_prob = jnp.array(0.0, dtype=jnp.float32)

        log_prior = self.sum_logpriors()
        log_prob += log_prior

        log_lik = self.digraph.nodes["response"]["attr"].get("log_lik", 0.0)
        log_lik = jnp.sum(log_lik)
        log_prob += log_lik

        return log_prob

    def build_graph_recursive(self,
                              obj: Any,
                              child_node: str,
                              param_name: str,
                              params: dict) -> None:
        """
        Method to build the DAG recursively. Calls methods add_*() to create the
        different node types. Note that add_root() is called outside to define the root
        of the DAG.

        Args:
            obj (Any): Classes Hyper, Param and Lpred of tigerpy.model.model.
            child_node (str): Name of the child node.
            param_name (str): Name of the current node/Parameter name in a distribution.
            params (dict): Dictionary that gets filled with the initial supplied
            values.
        """

        if isinstance(obj, Lpred):
            node_name = param_name
            self.add_lpred_node(name=node_name, 
                                lpred=obj)
            self.digraph.add_edge(node_name, 
                                  child_node, 
                                  role=param_name,
                                  bijector=obj.function)
            name_fixed = obj.obs.name
            self.add_fixed_node(name=name_fixed, 
                                obs=obj.obs)
            self.digraph.add_edge(name_fixed, 
                                  node_name, 
                                  role="fixed")
        elif isinstance(obj, Param):
            node_name = obj.name
            self.add_strong_node(name=node_name, 
                                 param=obj)
            self.digraph.add_edge(node_name, 
                                  child_node, 
                                  role=param_name)
            params[node_name] = obj.value
            param_name = node_name
        elif isinstance(obj, Hyper):
            node_name = obj.name
            self.add_hyper_node(name=node_name, 
                                hyper=obj)
            self.digraph.add_edge(node_name, 
                                  child_node, 
                                  role=param_name)
            param_name = node_name

        if isinstance(obj, (list, tuple)):
            for item in obj:
                self.build_graph_recursive(obj=item,
                                           child_node=child_node,
                                           param_name=param_name,
                                           params=params)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                self.build_graph_recursive(obj=value,
                                           child_node=param_name,
                                           param_name=key,
                                           params=params)
        elif hasattr(obj, "__dict__"):
            for value in obj.__dict__.values():
                self.build_graph_recursive(obj=value,
                                           child_node=child_node,
                                           param_name=param_name,
                                           params=params)

    def init_graph(self, params: dict) -> None:
        """
        Method to initialize the DAG.

        Args:
            params (dict): Dictionary with the initial supplied values for the parameters 
            to initialize the DAG.
        """

        # Use the topological sort algorithm
        for node in self.traversal_order:
            node_type = self.digraph.nodes[node]["node_type"]
            attr = self.digraph.nodes[node]["attr"]
            childs = list(self.digraph.successors(node))

            # Currently Graph structure only allows for one sucessor.
            if childs:
                # Select first successor and neglect all other
                child = childs[0]

            if node_type == "hyper":
                input_pass = attr["value"]

                edge = self.digraph.get_edge_data(node, child)
                self.digraph.nodes[child]["input"][edge["role"]] = input_pass

            elif node_type == "fixed":
                input_pass = attr["value"]

                edge = self.digraph.get_edge_data(node, child)
                self.digraph.nodes[child]["input"][edge["role"]] = input_pass

            elif node_type == "lpred":
                attr["value"] = self.update_lpred(node, attr)
                input_pass = attr["value"]

                edge = self.digraph.get_edge_data(node, child)
                self.digraph.nodes[child]["input"][edge["role"]] = input_pass

            elif node_type == "strong":
                attr["value"] = params[node]
                
                # Store to which parameter of the response the strong node belongs
                param_response = self.response_param_member(node)
                attr["param_response"] = param_response

                for parent in self.digraph.predecessors(node):
                    if self.digraph.nodes[parent]["node_type"] not in ["hyper", "fixed"]:
                        self.digraph.nodes[node]["input_fixed"] = False
                        break

                # If all inputs are fixed we need to initialize the probabiltiy distribution only onces.
                if self.digraph.nodes[node]["input_fixed"]:
                    attr["dist"] = self.init_dist(dist=attr["dist"],
                                                  params=self.digraph.nodes[node]["input"])
                else:
                    init_dist = self.init_dist(dist=attr["dist"],
                                               params=self.digraph.nodes[node]["input"])
                    attr["log_prior"] = self.logprior(init_dist, attr["value"])

                input_pass = attr["value"]
                edge = self.digraph.get_edge_data(node, child)
                self.digraph.nodes[child]["input"][edge["role"]] = input_pass

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

        # Build DAG by searching recursively through the classes
        self.build_graph_recursive(obj=self.model.response_dist,
                                   child_node="response",
                                   param_name="response",
                                   params=params)

        # Obtain the traversal order for the initialization of the graph
        self.init_traversal()

        # Obtain the traversal order to update the Graph i.e. for all probabilistic nodes and linear predictor nodes
        self.init_update_traversal()

        # Obtain the traversal order for all probabilistic nodes
        self.init_prob_traversal()

        # Initialize the full graph once
        self.init_graph(params)

        # store the values of the parameters 
        self.params = params

    def response_param_member(self, 
                              node: str) -> str:
        """
        Method to obtain the information to which parameter of the response the 
        paramter block belongs. 

        Args:
            node (str): Name of the node. 

        Returns:
            str: Return the parameter of the response to which the parameter
            block in the model belongs to.
        """


        childs = list(self.digraph.successors(node))
        if childs:
            child = childs[0]
            edge = self.digraph.get_edge_data(node, child)
            param = edge["role"]
        else: 
            child = node
            param = None

        if param in ["scale"]:
            return param
        elif self.digraph.nodes[child]["node_type"] != "root":
            return self.response_param_member(child)
        else: 
            return param

    def visualize_graph(self, savepath: Union[None, str]=None) -> None:
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

        def calculate_font_size(graph):
            num_nodes = len(graph.nodes)
            base_font_size = 12  # Initial font size
            scaling_factor = 0.25  # Adjust this factor based on your preference
            return round(base_font_size - scaling_factor * num_nodes)
        
        def calculate_node_size(graph):
            num_nodes = len(graph.nodes)
            base_font_size = 1000  # Initial font size
            scaling_factor = 5  # Adjust this factor based on your preference
            return base_font_size - scaling_factor * num_nodes

        node_labels = {node: label for node, label in labels.items() if node in node_type}

        strong_nodes = [node for node, node_type in node_type.items() if node_type == "strong" or node_type == "root"]
        weak_nodes = [node for node, node_type in node_type.items() if node_type != "strong"]

        nx.draw_networkx_edges(self.digraph, 
                               pos, 
                               ax=ax, 
                               node_size=calculate_node_size(self.digraph))
        nx.draw_networkx_labels(self.digraph, 
                                pos, 
                                ax=ax, 
                                labels=node_labels, 
                                font_size=calculate_font_size(self.digraph))
        nx.draw_networkx_nodes(self.digraph, 
                               pos, 
                               nodelist=strong_nodes, 
                               ax=ax, 
                               node_color="lightblue",
                               node_shape=node_shapes["strong"], 
                               node_size=calculate_node_size(self.digraph))
        nx.draw_networkx_nodes(self.digraph, 
                               pos, 
                               nodelist=weak_nodes, 
                               ax=ax, 
                               node_color="lightblue",
                               node_shape=node_shapes["weak"], 
                               node_size=calculate_node_size(self.digraph))

        edge_labels = nx.get_edge_attributes(self.digraph, "bijector")

        for key, value in edge_labels.items():
            if value is not None:
                string_to_search = str(edge_labels[key])
                pattern = r"jax\.numpy\.(\w+)"
                match_pattern = re.search(pattern, string_to_search)
                edge_labels[key] = match_pattern.group()
            else:
                edge_labels[key] = ""

        nx.draw_networkx_edge_labels(self.digraph, 
                                     pos, 
                                     edge_labels=edge_labels, 
                                     ax=ax, 
                                     font_size=8)

        if savepath is not None:
            plt.savefig(savepath)