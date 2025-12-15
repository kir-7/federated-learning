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
import math
from collections import defaultdict

class FlowerStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_clients: int,
        initial_parameters: Parameters,
        k_neighbours:float = 0.4,
        sigma_threshold: float = 1.0,
        fraction_fit: float = 1.0, # Usually 1.0 for this graph approach (active clients)
        fraction_evaluate:float =1.0,
        global_dataset=None,
        global_bs=None,
        evaluate_frequency:int=5,
        total_rounds:int=30,
        start_knn:int=5,
    ):
        self.num_clients = num_clients
        self.sigma_threshold = sigma_threshold
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.k_neighbours = k_neighbours
        self.start_knn = start_knn
        self.evaluate_freq = evaluate_frequency
        self.total_rounds = total_rounds
 
        self.topk = max(1, int(k_neighbours * num_clients))
       
        if global_dataset and global_bs:
            self.global_dataset = global_dataset
            self.global_loader = DataLoader(global_dataset, batch_size=128, shuffle=True)
        else:
            self.global_dataset, self.global_loader = None, None
        
        self.weights = parameters_to_ndarrays(initial_parameters)

        self.client_cids = set()
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
            else:
                client_parameters = ndarrays_to_parameters(self.weights)
                self.client_cids.add(client.cid)
                self.client_models[client.cid] = self.weights 

            fitins = fl.common.FitIns(client_parameters, {"server_round":server_round})
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
        client_weights = {}
        to_aggregate = defaultdict(list)

        for client, fit_res in results:
            w = parameters_to_ndarrays(fit_res.parameters)
            client_weights[client.cid] = (w, fit_res.num_examples)
            loss_aggregated.append(fit_res.metrics['train_loss'])
            self.client_models[client.cid] = w

        active_cids = list(client_weights.keys())

        distances = self.calculate_pairwise_distances(client_weights, active_cids)
        similarities = self.get_topk_similarities(server_round, distances, active_cids)
        
        for node_cid in active_cids:
            for nei_cid in active_cids:
                sim = similarities.get((node_cid, nei_cid), 0.0)
                w_nei, n_nei = client_weights[nei_cid]

                if sim > 0:
                    to_aggregate[node_cid].append((w_nei, n_nei, sim))
        
        for client_cid, neighbors in to_aggregate.items():
            self.client_models[client_cid] = self.similarity_aggregate(neighbors)
        
        metrics = {"avg_train_loss": sum(loss_aggregated) / len(loss_aggregated), "similarity_scores": similarities}

        return ndarrays_to_parameters(self.client_models[active_cids[0]]), metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateIns]]:

        if server_round==1 or server_round % self.evaluate_freq == 0 or server_round == self.total_rounds:
            
            # Sample clients for evaluation
            sample_size = int(self.num_clients * self.fraction_evaluate)
            clients = client_manager.sample(sample_size, min_num_clients=1)

            evaluate_configurations = []

            for client in clients :
                if client.cid in self.client_models:
                    client_parameters = ndarrays_to_parameters(self.client_models[client.cid])                
                else:
                    client_parameters = ndarrays_to_parameters(self.weights)
                    self.client_cids.add(client.cid)
                    self.client_models[client.cid] = self.weights

                evaluate_ins = EvaluateIns(client_parameters, {"server_round":server_round})
                evaluate_configurations.append((client, evaluate_ins))

            return evaluate_configurations
        
        else:
            return []
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
        precisions = [r.metrics["macro_precision"] * r.num_examples for _, r in results]
        recalls = [r.metrics["macro_recall"] * r.num_examples for _, r in results]
        f1_scores = [r.metrics["macro_f1"] * r.num_examples for _, r in results]
        
        examples = [r.num_examples for _, r in results]
        losses = [r.metrics['val_loss'] * r.num_examples for _, r in results]

        total_examples = sum(examples)
        if total_examples == 0:
            weighted_acc = 0
            weighted_loss = 0
        else:
            weighted_acc = sum(accuracies) / total_examples
            weighted_loss = sum(losses) / total_examples
            weighted_pr = sum(precisions) / total_examples
            weighted_re = sum(recalls) / total_examples
            weighted_f1 = sum(f1_scores) / total_examples

        clear_output(wait=True)
        print(f"Round {server_round} - Average Accuracy of Personalized Models: {weighted_acc * 100:.2f}%\n Average Precision: {weighted_pr * 100:.2f}\n Average Recall: {weighted_re * 100:.2f}\n Average F1: {weighted_f1 * 100:.2f}")

        return float(weighted_loss), {"accuracy": float(weighted_acc), "precision":float(weighted_pr), "recall":float(weighted_re), "f1_score":float(weighted_f1)}


    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, float]]]:
        """
        Evaluate the global centroid model on server-side data (if available).
        To use this, pass an 'evaluate_fn' to the Strategy if you want custom logic,
        otherwise, return None implies no centralized evaluation.
        """       
        return None      
    
    def flatten(self, weights_ndarrays : NDArrays):
        return np.hstack([layer.flatten() for layer in weights_ndarrays])
    
    def get_topk_similarities(self, server_round, distances, activate_cids):
        
        valid_dists = [v for v in distances.values() if v > 0]
        if not valid_dists:
            sigma = 1.0
        else:
            sigma = np.quantile(valid_dists, 0.5)
            if sigma == 0: sigma = 1.0 

        similarities = {}

        for client_a in activate_cids:
            candidates = []
            for client_b in activate_cids:
                candidates.append((client_b, math.exp(-distances[(client_a, client_b)]/sigma)))
            
            candidates.sort(key=lambda x:x[1], reverse=True)

            for i, (nei, similarity) in enumerate(candidates):
                if  server_round < self.start_knn:
                    similarities[(client_a, nei)] = similarity
                elif i <= self.topk:
                    similarities[(client_a, nei)] = similarity
                else:
                    similarities[(client_a, nei)] = 0
        
        return similarities

    def calculate_pairwise_distances(self, client_weights : Dict[int, Tuple[NDArrays, int]], active_cids) -> Dict[Tuple[int, int], float]:

        distances = {}
        flat_vecs = {cid: self.flatten(client_weights[cid][0]) for cid in active_cids}

        for client_a in active_cids:
            for client_b in active_cids:
                if client_a == client_b:
                    distances[(client_a, client_b)] = 0.0
                else:
                    dist = np.linalg.norm(flat_vecs[client_a] - flat_vecs[client_b], ord=2)
                    distances[(client_a, client_b)] = float(dist)

        return distances
    
    def similarity_aggregate(self, client_neighbours : list[tuple[NDArrays, int, float]]) -> NDArrays:    

        sum_aggregation_weights = sum(num_examples * similarity for (_, num_examples, similarity) in client_neighbours) 

        weighted_weights = [
            [layer * num_examples * similarity for layer in weights] for weights, num_examples, similarity in client_neighbours
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / sum_aggregation_weights
            for layer_updates in zip(*weighted_weights, strict=True)
        ]
        return weights_prime