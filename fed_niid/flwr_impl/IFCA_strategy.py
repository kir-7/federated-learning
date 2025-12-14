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
import copy

class FlowerStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_clients: int,
        initial_parameters: Parameters,
        k_clusters:int = 3,
        fraction_fit: float = 1.0, # Usually 1.0 for this graph approach (active clients)
        fraction_evaluate:float =1.0,
        global_dataset=None,
        global_bs=None,
        model_layer_counts:int=None
    ):
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.k_clusters = k_clusters

        assert model_layer_counts is not None, "Model Layer counts need to be passes"

        self.model_layer_counts = model_layer_counts
                
        if global_dataset and global_bs:
            self.global_dataset = global_dataset
            self.global_loader = DataLoader(global_dataset, batch_size=128, shuffle=True)
        else:
            self.global_dataset, self.global_loader = None, None
        
        self.weights = parameters_to_ndarrays(initial_parameters)

        self.cluster_models = {i:copy.deepcopy(self.weights) for i in range(k_clusters)}
        
    def initialize_parameters(self, client_manager):
        initial_parameters = ndarrays_to_parameters(self.weights)        
        return initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:

        sample_size = int(self.num_clients * self.fraction_fit)
        clients = client_manager.sample(sample_size, min_num_clients=1)

        fit_configurations = []

        stacked_weights = []
        for i in range(self.k_clusters):
            stacked_weights.extend(self.cluster_models[i])
        
        stacked_parameters = ndarrays_to_parameters(stacked_weights)

        for client in clients :
            # pass all the cluster models
            fitins = fl.common.FitIns(stacked_parameters, {"server_round":server_round, "k_clusters":self.k_clusters, "layers_per_model":self.model_layer_counts})
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
        cluster_to_clients = defaultdict(list)
        
        for client, fit_res in results:
            w = parameters_to_ndarrays(fit_res.parameters)            
            loss_aggregated.append(fit_res.metrics['train_loss'])
            cluster_to_clients[fit_res.metrics['assigned_cluster']].append((client.cid, fit_res.num_examples, w))
            

        for i in range(self.k_clusters):
            if i in cluster_to_clients:
                self.cluster_models[i] = self.cluster_aggregate(cluster_to_clients[i])
            else:
                pass 

        metrics = {"avg_train_loss": sum(loss_aggregated) / len(loss_aggregated)}

        return ndarrays_to_parameters(self.cluster_models[0]), metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateIns]]:

        # Sample clients for evaluation
        sample_size = int(self.num_clients * self.fraction_evaluate)
        clients = client_manager.sample(sample_size, min_num_clients=1)

        evaluate_configurations = []

        stacked_weights = []
        for i in range(self.k_clusters):
            stacked_weights.extend(self.cluster_models[i])
        
        stacked_parameters = ndarrays_to_parameters(stacked_weights)

        for client in clients :
            # pass all the cluster models
            evaluate_ins = EvaluateIns(stacked_parameters, {"server_round":server_round, "k_clusters":self.k_clusters, "layers_per_model":self.model_layer_counts})
            evaluate_configurations.append((client, evaluate_ins))

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
        
        cluster_assignments = defaultdict(list)
        for client, eval_res in results:
            cluster_assignments[eval_res.metrics['assigned_cluster']].append(client.cid)
       
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

        return float(weighted_loss), {"accuracy": float(weighted_acc), "precision":float(weighted_pr), "recall":float(weighted_re), "f1_score":float(weighted_f1), "cluster_assignments":cluster_assignments}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, float]]]:
        """
        Evaluate the global centroid model on server-side data (if available).
        To use this, pass an 'evaluate_fn' to the Strategy if you want custom logic,
        otherwise, return None implies no centralized evaluation.
        """       
        return None      
    
    def cluster_aggregate(self, clients_weights : List[tuple[str, int, NDArrays]]) -> NDArrays:    

        num_clients = len(clients_weights)

        weighted_weights = [
            [layer  for layer in weights] for _, _, weights in clients_weights
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_clients
            for layer_updates in zip(*weighted_weights, strict=True)
        ]
        return weights_prime