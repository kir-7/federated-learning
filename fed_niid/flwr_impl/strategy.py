import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    EvaluateIns,
    EvaluateRes,
    Code,
)
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
import torch
import copy

class FedGraphStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_clients: int,
        initial_parameters: Parameters,
        k_neighbours: float = 0.4,
        sigma_threshold: float = 1.0,
        fraction_fit: float = 1.0, # Usually 1.0 for this graph approach (active clients)
    ):
        self.num_clients = num_clients
        self.k_neighbours = k_neighbours
        self.sigma_threshold = sigma_threshold
        self.fraction_fit = fraction_fit
        
        # Initialize internal registry: {client_id_str: list_of_np_arrays}
        # We start by giving everyone the same initial weights
        initial_weights = parameters_to_ndarrays(initial_parameters)
        self.model_registry = {str(i): copy.deepcopy(initial_weights) for i in range(num_clients)}

        
        # Store the graph matrix S_t
        self.S_t = None

    def initialize_parameters(self, client_manager):
        # We handle initialization in __init__, so we return None here
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        
        # 1. Select clients
        # In Graph FL, we typically want to sample a subset, but we need the graph of ALL clients.
        # Flower handles sampling here.
        sample_size = int(self.num_clients * self.fraction_fit)
        clients = client_manager.sample(sample_size, min_num_clients=1)

        fit_configurations = []

        for client in clients:
            cid = client.cid

            print(f"Available model registry: {self.model_registry.keys()}")
            print(f"Got cid: {cid}")
            
            # 2. Retrieve the PERSONALIZED model for this client from our registry
            # logic: Each client gets its own model (calculated in aggregate_fit last round)
            specific_weights = self.model_registry.get(cid)
            
            # Convert to Parameters
            specific_params = ndarrays_to_parameters(specific_weights)
            
            # Create config (pass round number or other hyperparameters)
            fit_config = {"round": server_round}

            fit_configurations.append((client, fl.common.FitIns(specific_params, fit_config)))

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        
        if not results:
            return None, {}

        # 1. Update Registry with training results
        # Only clients who were sampled and trained will update the registry. 
        # Others keep their old weights (stale).
        loss_aggregated = []
        
        for client, fit_res in results:
            cid = client.cid
            updated_weights = parameters_to_ndarrays(fit_res.parameters)
            self.model_registry[cid] = updated_weights
            loss_aggregated.append(fit_res.metrics.get("train_loss", 0.0))

        # 2. Build Graph & Perform Neighbor Averaging
        # This corresponds to your `build_graph` and `fed_avg` logic
        self._update_graph_and_average()

        # 3. Return a "Global" model for checkpointing (optional)
        # We can just return the average of all models in the registry
        avg_weights = self._simple_average(list(self.model_registry.values()))
        
        metrics = {"avg_train_loss": sum(loss_aggregated) / len(loss_aggregated)}
        
        return ndarrays_to_parameters(avg_weights), metrics
    
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateIns]]:
        
        # Sample clients for evaluation
        sample_size = int(self.num_clients * self.fraction_evaluate)
        clients = client_manager.sample(sample_size, min_num_clients=1)
        
        eval_configurations = []
        
        for client in clients:
            # CRITICAL: Send the client their specific PERSONALIZED model to evaluate.
            # Do NOT send 'parameters' (which is the global centroid).
            specific_weights = self.model_registry.get(client.cid)
            
            if specific_weights is None:
                # Fallback to global centroid if client specific model missing
                specific_weights = parameters_to_ndarrays(parameters)
                
            specific_params = ndarrays_to_parameters(specific_weights)
            eval_config = {"round": server_round}
            
            eval_configurations.append((client, EvaluateIns(specific_params, eval_config)))
            
        return eval_configurations

    # --------------------------------------------------------------------------
    # 4. Aggregate Evaluate (Calculate Average Performance)
    # --------------------------------------------------------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, float]]:
        
        if not results:
            return None, {}

        # Weigh accuracy by the number of test examples the client had
        accuracies = [r.metrics["val_acc"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        
        # Aggregate and calculate weighted average
        if sum(examples) == 0:
            weighted_acc = 0
        else:
            weighted_acc = sum(accuracies) / sum(examples)
            
        print(f"Round {server_round} - Average Accuracy of Personalized Models: {weighted_acc * 100:.2f}%")
        
        return float(weighted_acc), {"accuracy": float(weighted_acc)}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, float]]]:
        """
        Evaluate the global centroid model on server-side data (if available).
        To use this, pass an 'evaluate_fn' to the Strategy if you want custom logic,
        otherwise, return None implies no centralized evaluation.
        """
        return None
    
    # --- Graph Logic (Adapted from your code) ---
    
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