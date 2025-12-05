import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    EvaluateIns,
    EvaluateRes,
    NDArrays,
)

from typing import List, Tuple, Union, Optional, Dict
from functools import reduce
from torch.utils.data import DataLoader
from IPython.display import clear_output
import numpy as np
import copy
import math

class FlowerStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_clients: int,
        initial_parameters: Parameters,
        k_neighbours:float = 0.4,
        global_rounds: int = 10,
        sigma_threshold: float = 1.0,
        fraction_fit: float = 1.0, # Usually 1.0 for this graph approach (active clients)
        global_dataset=None,
        global_bs=None,
    ):
        self.num_clients = num_clients
        self.sigma_threshold = sigma_threshold
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_fit
        self.global_rounds = global_rounds
        self.k_neighbours = k_neighbours

        if fraction_fit < 1:
            raise ValueError(f"Currently not supported for paricipation ratio < 1")

        self.topk = max(1, int(k_neighbours * num_clients))
       
        if global_dataset and global_bs:
            self.global_dataset = global_dataset
            self.global_loader = DataLoader(global_dataset, batch_size=128, shuffle=True)
        else:
            self.global_dataset, self.global_loader = None, None
        
        self.weights = parameters_to_ndarrays(initial_parameters)

        self.client_cids = []
        self.client_models = {}
        
    def initialize_parameters(self, client_manager):
        initial_parameters = ndarrays_to_parameters(self.weights)        
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:

        sample_size = int(self.num_clients * self.fraction_fit)
        clients = client_manager.sample(sample_size, min_num_clients=1)

        fit_configurations = []

        for client in clients :
            if client.cid in self.client_models:
                client_parameters = ndarrays_to_parameters(self.client_models[client.cid])
                fitins = fl.common.FitIns(client_parameters, {"server_round":server_round})
                fit_configurations.append((client, fitins))
            else:
                client_parameters = ndarrays_to_parameters(self.weights)
                fitins = fl.common.FitIns(client_parameters, {"server_round":server_round})
                fit_configurations.append((client, fitins))
                self.client_cids.append(client.cid)
                self.client_models[client.cid] = self.weights 

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
        client_weights = {}
        to_aggregate = {}

        for client, fit_res in results:
            client_weights[client.cid] = (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            loss_aggregated.append(fit_res.metrics['train_loss'])

        distances = self.calculate_pairwise_distances(client_weights)
        similarities = {}

        sigma = np.quantile([v for _, v in distances.items() if v > 0], 0.5)

        # get topk neighbours, distance should be least and similarity should be high
        for i, (key, dist) in enumerate(sorted(distances.items(), key=lambda x: x[1], reverse=True)):
            if i > self.topk:
                similarities[key] = 0
            else:
                similarities[key] = math.exp(-dist/sigma)
        
        for node_cid in self.client_models.keys():
            for nei_cid in self.client_models.keys():
                to_aggregate[node_cid].append(client_weights[nei_cid] + (similarities[(node_cid, nei_cid)], ))
        
        self.client_models = {client_cid:self.distance_aggregate(client_neighbours) for client_cid, client_neighbours in to_aggregate.items()}

        metrics = {"avg_train_loss": sum(loss_aggregated) / len(loss_aggregated), "similarity_scores": similarities}

        return ndarrays_to_parameters(self.client_models[self.client_cids[0]]), metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateIns]]:

        # Sample clients for evaluation
        sample_size = int(self.num_clients * self.fraction_evaluate)
        clients = client_manager.sample(sample_size, min_num_clients=1)

        evaluate_configurations = []

        for client in clients :
            if client.cid in self.client_models:
                client_parameters = ndarrays_to_parameters(self.client_models[client.cid])
                evaluate_ins = EvaluateIns(client_parameters, {"server_round":server_round})
                evaluate_configurations.append((client, evaluate_ins))
            else:
                client_parameters = ndarrays_to_parameters(self.weights)
                evaluate_ins = EvaluateIns(client_parameters, {"server_round":server_round})
                evaluate_configurations.append((client, evaluate_ins))
                self.client_cids.append(client.cid)
                self.client_models[client.cid] = self.weights

        return evaluate_configurations

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
        losses = [r.metrics['val_loss'] * r.num_examples for _, r in results]


        if sum(examples) == 0:
            weighted_acc = 0
            weighted_loss = 0
        else:
            weighted_acc = sum(accuracies) / sum(examples)
            weighted_loss = sum(losses) / sum(examples)

        clear_output(wait=True)
        print(f"Round {server_round} - Average Accuracy of Personalized Models: {weighted_acc * 100:.2f}%")

        return float(weighted_loss), {"accuracy": float(weighted_acc)}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, float]]]:
        """
        Evaluate the global centroid model on server-side data (if available).
        To use this, pass an 'evaluate_fn' to the Strategy if you want custom logic,
        otherwise, return None implies no centralized evaluation.
        """       
        return None      
    
    def flatten(self, weights_ndarrays : NDArrays):
        return np.hstack([layer.flatten() for layer in weights_ndarrays])
    
    def calculate_pairwise_distances(self, client_weights : Dict[int, Tuple[NDArrays, int]]) -> Dict[Tuple[int, int], float]:

        distances = {}

        for client_a in self.client_cids:
            for client_b in self.client_cids:
                vec_a, vec_b = self.flatten(client_weights[client_a][0]), self.flatten(client_weights[client_b][0])
                distance = np.linalg.norm(vec_a-vec_b, ord=2)
                distances[(client_a, client_b)] = distance

        return distances
    
    def distance_aggregate(self, client_neighbours : list[tuple[NDArrays, int, float]]) -> NDArrays:
        
        num_examples_total = sum(num_examples for (_, num_examples, _) in client_neighbours)
        distance_total = sum(distance for (_, _, distance) in client_neighbours)


        weighted_weights = [
            [layer * num_examples * distance for layer in weights] for weights, num_examples, distance in client_neighbours
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / (num_examples_total*distance_total)
            for layer_updates in zip(*weighted_weights, strict=True)
        ]
        return weights_prime