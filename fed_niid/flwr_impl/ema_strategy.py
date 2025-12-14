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
from flwr.server.strategy.aggregate import aggregate

from typing import List, Tuple, Union, Optional, Dict
from functools import reduce
from torch.utils.data import DataLoader
from IPython.display import clear_output
import math

class FlowerStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_clients: int,
        initial_parameters: Parameters,
        global_rounds: int = 10,
        sigma_threshold: float = 1.0,
        fraction_fit: float = 1.0, # Usually 1.0 for this graph approach (active clients)
        fraction_evaluate: float = 1.0,
        global_dataset=None,
        global_bs=None,
        ema_alpha=0.8,
        anneal_alpha=False
    ):
        self.num_clients = num_clients
        self.sigma_threshold = sigma_threshold
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.global_rounds = global_rounds

        if global_dataset and global_bs:
            self.global_dataset = global_dataset
            self.global_loader = DataLoader(global_dataset, batch_size=128, shuffle=True)
        else:
            self.global_dataset, self.global_loader = None, None

        self.ema_alpha = ema_alpha
        self.anneal_alpha = anneal_alpha
        self.decay_rate = -((math.log(0.7/self.ema_alpha)) / max(1, self.global_rounds))
        
        self.weights = parameters_to_ndarrays(initial_parameters)

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
            fitins = fl.common.FitIns(parameters, {"server_round":server_round})
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

        # a simple ema based aggregation instead of fed prox
        aggregated_ndarrays = aggregate(to_aggregate)
        
        alpha = self.ema_alpha if not self.anneal_alpha else self.get_ema_alpha(server_round)
        
        if self.anneal_alpha:
            print(f"Round {server_round}: Annealing EMA Alpha to {alpha:.4f}") 

        smoothed_ndarrays = self.smooth_ndarrays(alpha, aggregated_ndarrays)

        self.weights = smoothed_ndarrays
        parameters_aggregated = ndarrays_to_parameters(smoothed_ndarrays)

        metrics = {"avg_train_loss": sum(loss_aggregated) / len(loss_aggregated)}

        return parameters_aggregated, metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateIns]]:

        # Sample clients for evaluation
        sample_size = int(self.num_clients * self.fraction_evaluate)
        clients = client_manager.sample(sample_size, min_num_clients=1)

        eval_configurations = []
        evaluate_ins = EvaluateIns(parameters, {"server_round":server_round})

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
    
    
    def get_ema_alpha(self, round_num):
        return self.ema_alpha * math.exp(- self.decay_rate * round_num)

    def smooth_ndarrays(self, alpha, aggregated_ndarrays):
               
        # Compute smoothed weights for each layer
        weights_prime: NDArrays = []

        for aggregate_layer, prev_layer in zip(aggregated_ndarrays, self.weights):
            layer_prime = (alpha * aggregate_layer) + ((1-alpha) * prev_layer)
            weights_prime.append(layer_prime)

        return weights_prime 