import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
import numpy as np
import math
import time

from dataclasses import dataclass

from dataset import FedDataset
from sampling import mnist_noniid_sample, mnist_noniid_sample_consistent, plot_client_distributions
from viz import FederatedLearningVisualizer

class FedAVGClient:
    def __init__(self, data_partition, client_id, transform, epochs=5, lr=1e-3, bs=32):

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        print(f"Client {client_id} initialized using {self.device}")

        self.client_id = client_id
        self.epochs = epochs
        self.lr = lr
        self.bs = bs

        self.data_partition_train, self.data_partition_val = data_partition
        self.train_loader, self.val_loader = self.split_train_val(transform)
        
        self.distribution_stats = self.get_distribution_stats()

    def normalize_distribution(self, count_dict):
        total = sum(count_dict.values())
        return {k: v/total for k, v in count_dict.items()}

    def get_distribution_stats(self):
        return self.normalize_distribution({k:len(v) for k, v in self.data_partition_train.items()})

    @staticmethod
    def create_dataset_from_samples(data_partition_train, data_partition_val, transform):        
        # logic to create torch train and val dataset from data partition : format  {class:list[smaples]}

        dataset_train = FedDataset([(img, k) for k, v in data_partition_train.items() for img in v], transform=transform)   
        dataset_val = FedDataset([(img, k) for k, v in data_partition_val.items() for img in v], transform=transform)   
    
        return dataset_train, dataset_val

    def split_train_val(self, transform):
        
        train_dataset, val_dataset = self.create_dataset_from_samples(self.data_partition_train, self.data_partition_val, transform)
        train_loader = DataLoader(train_dataset, batch_size=self.bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.bs, shuffle=False)
        return train_loader, val_loader


    def update(self, model, global_round):

        
        model.train()
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)        
        
        epoch_loss = []
        
        print(f"\nClient {self.client_id} starting training for round {global_round+1}")
        for epoch in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                model.zero_grad()
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())
                
            avg_loss = sum(batch_loss)/len(batch_loss) if batch_loss else 0
            epoch_loss.append(avg_loss)
            
            print(f"Client {self.client_id} - Round {global_round+1}, Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return model.state_dict(), sum(epoch_loss)/len(epoch_loss) if epoch_loss else 0
    
    def evaluate(self, model):
        
        model.eval()
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = test_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        print(f"Client {self.client_id} test - Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

class FedAVGServer:
    def __init__(self, model, sample_func, args, n_rounds=10, frac=1.0):
       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Server initialized using {self.device}")
        
        self.model = model.to(self.device)
        
        self.args = args
       
        self.data_partitions, validation_samples, transform, _ = sample_func(args.dataset_path, args.n_clients, alpha=args.alpha)          

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.val_loader = DataLoader(FedDataset([(img, k) for k, v in validation_samples.items() for img in v], transform=transform), batch_size=32, shuffle=False)
        self.visualizer = FederatedLearningVisualizer()

        self.n_rounds = n_rounds
        self.n_clients = args.n_clients   
        self.frac = frac      

        print(f"Creating {args.n_clients} clients")
        self.clients = []
        for i in range(args.n_clients):
            client = FedAVGClient(self.data_partitions[i], i, transform)
            self.clients.append(client)
            
        self.client_dist_stats = {k.client_id : k.distribution_stats for k in self.clients}
        
        print(f"Number of parameters: ", sum(p.numel() for p in model.parameters()))

        print("Jenson Shannon similarity between the two clients: ")
        similarity_mat = self.get_similarity_matrix()
        print(similarity_mat)
        
        self.visualizer.similarity_matrix = similarity_mat        

        print(f"plotting data distributions:")
        plot_client_distributions(self.data_partitions, save_path=f"mnist_plots/{args.name}.png")

        print(f"Server setup complete. Ready for {n_rounds} rounds of training")
    
    @staticmethod
    def kullback_liebler_div(dist_1, dist_2, epsilon=1e-8):
        assert len(dist_1) == len(dist_2)
        
        div = 0
        for k, v in dist_1.items():
            if v > 0:  
                div += v * math.log(v / (dist_2[k] + epsilon))
        return div

    @staticmethod
    def jensen_shannon_similarity(dist_1, dist_2):
        assert len(dist_1) == len(dist_2)
        
        m_dist = {}
        for k, v in dist_1.items():
            m_dist[k] = (v + dist_2[k])/2

        return math.exp(-(FedAVGServer.kullback_liebler_div(dist_1, m_dist) + FedAVGServer.kullback_liebler_div(dist_2, m_dist)) / 2)    

    def get_similarity_matrix(self):
        matrix = np.ones((self.n_clients, self.n_clients))

        for i in range(self.n_clients):
            for j in range(self.n_clients):
                matrix[i][j] = FedAVGServer.jensen_shannon_similarity(self.client_dist_stats[i], self.client_dist_stats[j])
        
        return matrix 

    @staticmethod
    def fed_avg(local_weights):

        print("Performing FedAvg aggregation of client weights")
    
        w_avg = copy.deepcopy(local_weights[0])
        
        for key in w_avg.keys():
            for i in range(1, len(local_weights)):
                w_avg[key] += local_weights[i][key]
            w_avg[key] = torch.div(w_avg[key], len(local_weights))
        
        return w_avg
    
    def fed_avg_diversity_weighted(self, local_weights):
        similarity_matrix = self.get_similarity_matrix()
        # Lower similarity = higher weight (more diverse clients get more influence)
        diversity_scores = 2.0 - np.mean(similarity_matrix, axis=1)  
        client_weights = diversity_scores / np.sum(diversity_scores)
        return self.aggregate_with_weights(local_weights, client_weights)

    def aggregate_with_weights(self, local_weights, client_weights):
        w_avg = {}
        for key in local_weights[0].keys():
            w_avg[key] = torch.zeros_like(local_weights[0][key])
        
        # Perform weighted aggregation
        for i, weight in enumerate(client_weights):
            for key in w_avg.keys():
                w_avg[key] += local_weights[i][key] * weight
        
        return w_avg
    
    def fed_avg_sim_weighted_softmax(self, local_weights, temperature=2.0):
        similarity_matrix = self.get_similarity_matrix()
        avg_similarities = np.mean(similarity_matrix, axis=1)
        
        # Apply softmax with high temperature (more uniform)
        exp_similarities = np.exp(avg_similarities / temperature)
        client_weights = exp_similarities / np.sum(exp_similarities)
        
        # Ensure minimum weight threshold
        min_weight = 1.0 / (self.n_clients * 3)  # No client gets less than 1/3 of uniform weight
        client_weights = np.maximum(client_weights, min_weight)
        client_weights = client_weights / np.sum(client_weights)
        w_avg = {}
        for key in local_weights[0].keys():
            w_avg[key] = torch.zeros_like(local_weights[0][key])
        
        # Perform weighted aggregation
        for i, weight in enumerate(client_weights):
            for key in w_avg.keys():
                w_avg[key] += local_weights[i][key] * weight
        
        return w_avg

    def fed_avg_sim_weighted(self, local_weights):
        
        # Get the similarity matrix (n_clients x n_clients)
        similarity_matrix = self.get_similarity_matrix()
        
            # Weight each client by its average similarity to all others
        # This gives more weight to clients that are "representative" of the overall distribution
        client_weights = np.mean(similarity_matrix, axis=1)  # Average similarity for each client
        
        # Normalize weights so they sum to 1
        client_weights = client_weights / np.sum(client_weights)        
        
        # Initialize the averaged weights
        w_avg = {}
        for key in local_weights[0].keys():
            w_avg[key] = torch.zeros_like(local_weights[0][key])
        
        # Perform weighted aggregation
        for i, weight in enumerate(client_weights):
            for key in w_avg.keys():
                w_avg[key] += local_weights[i][key] * weight
        
        return w_avg

    @staticmethod
    def flatten_state_dict(state_dict):
        """Flatten all parameters in state_dict to a 1D vector"""
        flattened = []
        for param in state_dict.values():
            flattened.extend(param.flatten().cpu().numpy())
        return np.array(flattened)
    
    def get_distance_matrix(self, local_weights):
        model_states = np.array([self.flatten_state_dict(model_state) for model_state in local_weights])
        norms = np.linalg.norm(model_states, axis=1)
        
        distances = np.zeros((self.n_clients, self.n_clients))

        for i, v1 in enumerate(model_states):
            for j, v2 in enumerate(model_states):

                distances[i][j] = np.linalg.norm(v1-v2)/(norms[i] + norms[j] + 1e-8)

        return distances

    def evaluate(self):        
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                test_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = test_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
        return accuracy, avg_loss
    
    def train(self):
        """
        Execute federated learning training with clean, structured logging. 
        """       
        
        # Initialize tracking variables
        start_time = time.time()
        
        # Metrics tracking
        metrics = {
            'global_accuracies': [],
            'global_losses': [],
            'local_accuracy_averages': [],
            'local_loss_averages': [],            
        }        

        # Training loop
        for round_idx in range(self.n_rounds):
            
            # Header for each round
            print(f"\nROUND {round_idx + 1}/{self.n_rounds}")
            print("-" * 40)
            
            m = max(int(self.frac * self.n_clients), 1)
            selected_clients = np.random.choice(range(self.n_clients), m, replace=False)
            self.model.train()            

            # Local training phase
            print(f"\n📚 LOCAL TRAINING PHASE:")
            local_weights = []
            local_losses = []

            self.visualizer.add_global_weights(0, copy.deepcopy(self.model.state_dict()))
            
            for client_idx in selected_clients:
                print(f"  Training client {client_idx}...", end=" ")                                
                w, loss = self.clients[client_idx].update(copy.deepcopy(self.model), round_idx)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
                print(f"Loss: {loss:.4f}")
            
            # Aggregation phase

            self.visualizer.add_round_weights(round_idx+1, {i:v for i, v in enumerate(local_weights)})
            
            print(f"\n🔄 AGGREGATION PHASE:")
            print(f" Norm Distances between pair of clients")
            print(self.get_distance_matrix(local_weights))

            global_weights = self.fed_avg_diversity_weighted(local_weights)
            
            self.model.load_state_dict(global_weights)
            
            self.visualizer.add_global_weights(round_idx+1, copy.deepcopy(self.model.state_dict()))

            # Global evaluation            
            print(f"\n📊 EVALUATION PHASE:")
            
            # Local evaluation (update all clients with global model)
            local_accuracies = []
            local_losses_eval = []
            
            self.model.eval()

            print(f"  Client evaluations:")
            for idx, client in enumerate(self.clients):
                acc, loss = client.evaluate(copy.deepcopy(self.model))
                local_accuracies.append(acc)
                local_losses_eval.append(loss)
            
            # Calculate averages
            avg_local_acc = np.mean(local_accuracies)
            avg_local_loss = np.mean(local_losses_eval)
            avg_training_loss = np.mean(local_losses)
            
            global_acc, global_loss = self.evaluate()            

            # Store metrics
            metrics['global_accuracies'].append(global_acc)
            metrics['global_losses'].append(global_loss)
            metrics['local_accuracy_averages'].append(avg_local_acc)
            metrics['local_loss_averages'].append(avg_training_loss)
                        
            # Round summary
            print(f"\n📈 ROUND {round_idx + 1} SUMMARY:")
            print(f"  Global Model    → Acc: {global_acc:6.2f}% | Loss: {global_loss:.4f}")
            print(f"  Local Average   → Acc: {avg_local_acc:6.2f}% | Loss: {avg_local_loss:.4f}")
            print(f"  Training Loss   → Avg: {avg_training_loss:.4f}")
            
            # Progress indicator
            if round_idx < self.n_rounds - 1:
                print(f"\n{'=' * 40}")
        
        # Final summary
        print(f"\n{'=' * 80}")
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"📊 FINAL RESULTS:")
        print(f"  Final Global Acc  → {metrics['global_accuracies'][-1]:.2f}%")
        print(f"  Final Global Loss → {metrics['global_losses'][-1]:.4f}")
        print(f"  Final Local Acc   → {metrics['local_accuracy_averages'][-1]:.2f}%")
        print(f"  Final Local Loss  → {metrics['local_loss_averages'][-1]:.4f}")

        fig = self.visualizer.plot_weight_evolution_pca() 
        fig.savefig(f"mnist_plots/convergence_plot_{self.args.name}.png")
        
        fig_heatmap = self.visualizer.plot_convergence_heatmap(figsize=(5 * self.n_rounds, 4))
        fig_heatmap.savefig(f"mnist_plots/heatmap_{self.args.name}.png")

        return (
            self.visualizer,
            self.model.state_dict(),            
            metrics['global_losses'],
            metrics['global_accuracies'],
            metrics['local_loss_averages'],
            metrics['local_accuracy_averages']
        )

      
@dataclass
class TrainingArgs:
    dataset_path : str = "../data/mnist"
    n_clients : int = 3    
    alpha : float = 1000
    name : str = "3c_1000_alpha_div_5_lep_high_lr"

if __name__ == "__main__":

    cifar10_model = torch.nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='same'),
        nn.GELU(),
        nn.BatchNorm2d(32),

        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding='valid'),
        nn.GELU(),
        nn.BatchNorm2d(64),

        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding='valid'),
        nn.GELU(),
        nn.BatchNorm2d(128),
        
        nn.Conv2d(128, 64, kernel_size=3, stride=2, padding='valid'),
        nn.GELU(),
        nn.BatchNorm2d(64),
        
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(1),

        nn.Linear(64, 128),
        nn.GELU(),
        nn.Linear(128, 64),
        nn.GELU(),    
        nn.Linear(64, 32),
        nn.GELU(),

        nn.Linear(32, 10)
    )
    
    mnist_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    args = TrainingArgs()

    server = FedAVGServer(mnist_model, mnist_noniid_sample_consistent, args, n_rounds=10)
    server.train()

    # del server

    # args.alpha = 0.5
    # args.name = "3c_0-5_alpha_div"
    # server = FedAVGServer(mnist_model, mnist_noniid_sample_consistent, args, n_rounds=10)
    # server.train()

    # del server
  
