import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    EvaluateIns,
    EvaluateRes,
)
from flwr.server.strategy.aggregate import aggregate

from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
import torch
from IPython.display import clear_output

class FedGraphStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_clients: int,
        initial_parameters: Parameters,
        k_neighbours: float = 0.4,
        sigma_threshold: float = 1.0,
        fraction_fit: float = 1.0, 
    ):
        self.num_clients = num_clients
        self.k_neighbours = k_neighbours
        self.sigma_threshold = sigma_threshold
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_fit

        self.initial_weights = parameters_to_ndarrays(initial_parameters)
        # self.model_registry = {str(i): copy.deepcopy(initial_weights) for i in range(num_clients)}

        # Store the graph matrix S_t
        self.S_t = None

    def initialize_parameters(self, client_manager):
        # We handle initialization in __init__, so we return None here
        initial_parameters = self.initial_weights
        self.initial_weights = None
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:

        # 1. Select clients
        # In Graph FL, we typically want to sample a subset, but we need the graph of ALL clients.
        # Flower handles sampling here.
        sample_size = int(self.num_clients * self.fraction_fit)
        clients = client_manager.sample(sample_size, min_num_clients=1)

        fit_configurations = []

        for client in clients :
            fitins = fl.common.FitIns(parameters, {"round":server_round})
            fit_configurations.append((client, fitins))

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:

        if not results:
            return None, {}

        if failures:
            print(f"\n[!] Round {server_round} reported {len(failures)} failures.")
            for i, failure in enumerate(failures):
                # failure might be a tuple (ClientProxy, Exception) or just Exception
                if isinstance(failure, tuple):
                    proxy, exc = failure
                    print(f"    Failure {i} (Client {proxy.cid}): {exc}")
                else:
                    print(f"    Failure {i}: {failure}")

        loss_aggregated = []
        to_aggregate = []

        for _, fit_res in results:
            to_aggregate.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
            loss_aggregated.append(fit_res.metrics['train_loss'])

        aggregated_ndarrays = aggregate(to_aggregate)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics = {"avg_train_loss": sum(loss_aggregated) / len(loss_aggregated)}

        return parameters_aggregated, metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateIns]]:

        # Sample clients for evaluation
        sample_size = int(self.num_clients * self.fraction_evaluate)
        clients = client_manager.sample(sample_size, min_num_clients=1)

        eval_configurations = []
        evaluate_ins = EvaluateIns(parameters, {"round":server_round})

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, float]]:

        if not results:
            return None, {}

        if failures:
            print(f"Round {server_round} had {len(failures)} failures.")
            for fail in failures:
                # fail is a tuple or an exception depending on Flower version/context
                if isinstance(fail, BaseException):
                    print(f"Failure: {fail}")
                else:
                    # It's a tuple (client_proxy, exception)
                    print(f"Client {fail[0].cid} failed: {fail[1]}")


        accuracies = [r.metrics["val_acc"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        if sum(examples) == 0:
            weighted_acc = 0
        else:
            weighted_acc = sum(accuracies) / sum(examples)

        clear_output(wait=True)
        print(f"Round {server_round} - Average Accuracy of Personalized Models: {weighted_acc * 100:.2f}%")

        return float(weighted_acc), {"accuracy": float(weighted_acc)}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, float]]]:
        """
        Evaluate the global centroid model on server-side data (if available).
        To use this, pass an 'evaluate_fn' to the Strategy if you want custom logic,
        otherwise, return None implies no centralized evaluation.
        """
        return None

    def _flatten_params(self, weights_list):
        # Flatten a list of numpy arrays into a single 1D numpy array
        return np.concatenate([w.flatten() for w in weights_list])

    def _update_graph_and_average(self):
        """
        Replicates your build_graph + fed_avg logic using the Strategy's model registry.
        """
        # 1. Flatten all models for distance calculation
        # ensuring order 0..N-1
        sorted_cids = sorted(self.model_registry.keys(), key=lambda x: int(x))
        flat_models = np.stack([self._flatten_params(self.model_registry[cid]) for cid in sorted_cids])

        # Convert to torch for fast cdist (or stay in numpy)
        tensor_models = torch.tensor(flat_models)

        # 2. Compute Distance Matrix
        dist_mat = torch.cdist(tensor_models, tensor_models, p=2)

        # 3. Compute Similarity (Graph)
        mean_dist = dist_mat.mean()
        sigma = mean_dist if mean_dist > 0 else 1.0
        graph = torch.exp(-dist_mat / sigma) # Shape (N, N)

        # 4. Prune Edges (Top-K)
        k = max(2, int(self.num_clients * self.k_neighbours))
        vals, indices = torch.topk(graph, k=k, dim=1)
        new_graph = torch.zeros_like(graph)
        new_graph.scatter_(1, indices, vals)

        # Normalize rows
        row_sums = new_graph.sum(dim=1, keepdim=True)
        self.S_t = new_graph / (row_sums + 1e-8) # The adjacency matrix

        # 5. Perform Weighted Averaging (FedGraph Aggregation)
        # For every client, compute the weighted average of neighbors

        new_registry = {}

        # We iterate over clients to calculate their NEW model based on neighbors
        adjacency_np = self.S_t.cpu().numpy()

        for i, cid_target in enumerate(sorted_cids):
            neighbor_weights = adjacency_np[i] # Array of shape (N,)

            # Prepare to sum weights
            # Shape of a model: List[Array_Layer_1, Array_Layer_2, ...]
            # We need to perform linear combination per layer

            weighted_model_sum = [np.zeros_like(w) for w in self.model_registry[cid_target]]

            for j, cid_neighbor in enumerate(sorted_cids):
                w_ij = neighbor_weights[j]
                if w_ij > 0:
                    neighbor_model = self.model_registry[cid_neighbor]
                    for layer_idx, layer_w in enumerate(neighbor_model):
                        weighted_model_sum[layer_idx] += w_ij * layer_w

            new_registry[cid_target] = weighted_model_sum

        # Update the main registry with the graph-averaged models
        self.model_registry = new_registry

    def _simple_average(self, weights_lists):
        # Helper to average a list of models (for global checkpointing)
        num_models = len(weights_lists)
        avg = [np.zeros_like(w) for w in weights_lists[0]]
        for w_list in weights_lists:
            for i, w in enumerate(w_list):
                avg[i] += w
        return [w / num_models for w in avg]