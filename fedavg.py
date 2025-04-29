import torch
from torch import nn
from torch.utils.data import DataLoader
import copy
import numpy as np
from tqdm import tqdm
import time

class Client:
    def __init__(self, model, dataset, client_id, epochs=10, lr=0.01, bs=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        print(f"Client {client_id} initialized using {self.device}")

        self.model = copy.deepcopy(model).to(self.device)
        self.client_id = client_id
        self.epochs = epochs
        self.lr = lr
        self.bs = bs

        self.trainloader, self.valloader, self.testloader = self.train_val_test(dataset)
        self.criterion = nn.NLLLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

    def train_val_test(self, dataset):
        n_samples = len(dataset)
        train_data = dataset[:int(0.8*n_samples)]
        val_data = dataset[int(0.8*n_samples):int(0.9*n_samples)]
        test_data = dataset[int(0.9*n_samples):]

        trainloader = DataLoader(train_data,batch_size=self.bs, shuffle=True)
        valloader = DataLoader(val_data, batch_size=self.bs, shuffle=False) 
        testloader = DataLoader(test_data, batch_size=self.bs, shuffle=False)
        return trainloader, valloader, testloader
    
    def update(self, global_model_state_dict, global_round):
        
        self.model.load_state_dict(global_model_state_dict)
        
        self.model.train()
        epoch_loss = []
        
        print(f"\nClient {self.client_id} starting training for round {global_round+1}")
        for epoch in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                
                batch_loss.append(loss.item())
                
            avg_loss = sum(batch_loss)/len(batch_loss) if batch_loss else 0
            epoch_loss.append(avg_loss)
            print(f"Client {self.client_id} - Round {global_round+1}, Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return self.model.state_dict(), sum(epoch_loss)/len(epoch_loss) if epoch_loss else 0
    
    def evaluate(self):
        
        self.model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                test_loss += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total if total > 0 else 0
        avg_loss = test_loss / len(self.testloader) if len(self.testloader) > 0 else 0
        print(f"Client {self.client_id} test - Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        return accuracy, avg_loss

class Server:
    def __init__(self, model, sample, args, n_rounds=10, frac=1.0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Server initialized using {self.device}")
        
        # Initialize global model
        self.model = copy.deepcopy(model).to(self.device)
        print("Global model structure:")
        print(self.model)

        self.args = args        
        self.data_partition = sample(args)

        self.n_rounds = n_rounds
        self.frac = frac  

        print(f"Creating {args.n_clients} clients")
        self.clients = []
        for i in range(args.n_clients):
            client = Client(model, self.data_partition[i], i)
            self.clients.append(client)
            
        print(f"Server setup complete. Ready for {n_rounds} rounds of training")
    
    def fed_avg(self, local_weights):

        print("Performing FedAvg aggregation of client weights")
        
        w_avg = copy.deepcopy(local_weights[0])
        for key in w_avg.keys():
            # Sum weights from all clients
            for i in range(1, len(local_weights)):
                w_avg[key] += local_weights[i][key]
            # Compute average
            w_avg[key] = torch.div(w_avg[key], len(local_weights))
        return w_avg
    
    def train(self):

        print("\n=== Starting Federated Training (FedAvg) ===")
        start_time = time.time()
        
        global_weights = copy.deepcopy(self.model.state_dict())
        
        train_loss, train_accuracy = [], []
        
        for round_idx in tqdm(range(self.n_rounds), desc="Global Rounds"):
            local_weights, local_losses = [], []
            print(f"\n | Global Training Round: {round_idx+1}/{self.n_rounds} |")
            
            m = max(int(self.frac * self.n_clients), 1)
            selected_clients = np.random.choice(range(self.n_clients), m, replace=False)
            print(f"Selected {m} clients for this round: {selected_clients}")
            
            for client_idx in selected_clients:
                print(f"Training client {client_idx}")
                w, loss = self.clients[client_idx].update(global_weights, round_idx)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            
            global_weights = self.fed_avg(local_weights)
            
            self.model.load_state_dict(global_weights)
            
            loss_avg = sum(local_losses) / len(local_losses) if local_losses else 0
            train_loss.append(loss_avg)
            print(f"Round {round_idx+1} Average Loss: {loss_avg:.4f}")
            
            list_acc, list_loss = [], []
            for client in self.clients:
                client.model.load_state_dict(global_weights)
                acc, loss = client.evaluate()
                list_acc.append(acc)
                list_loss.append(loss)
            
            avg_acc = sum(list_acc) / len(list_acc) if list_acc else 0
            train_accuracy.append(avg_acc)
            print(f"Round {round_idx+1} Average Accuracy: {avg_acc:.2f}%")
        
        training_time = time.time() - start_time
        
        print(f"\n--- FedAvg Training Completed ---")
        print(f"Total Training Time: {training_time:.2f} seconds")
        print(f"Final Average Accuracy: {train_accuracy[-1]:.2f}%")
        print(f"Final Average Loss: {train_loss[-1]:.4f}")
        
        # Return final global model and metrics
        return self.model.state_dict(), train_loss, train_accuracy