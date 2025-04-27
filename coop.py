import torch
from torch import nn
from torch.utils.data import DataLoader
from copy import deepcopy
import threading
import math
import time
from tqdm.auto import tqdm
class Client:
    def __init__(self, model, dataset, client_id, get_model_age, get_bounds, get_model, update_server, epochs= 10, lr=3e-4, bs=32):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.client_id = client_id
        self.epochs = epochs
        self.split = 0.8
        self.lr = lr
        self.bs = bs
        self.trainloader, self.valloader = self.train_val(dataset)
        print(f"Client {client_id} training data: {len(self.trainloader.dataset)} samples")
        

        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.ak = 0

        self.get_server_age = get_model_age
        self.get_server_model = get_model
        self.get_bounds = get_bounds
        self.update_server = update_server

    def train_val(self,dataset):
        num_samples = len(dataset)
        train_samples = int(self.split*num_samples)

        train_loader = DataLoader(dataset[:train_samples], batch_size=self.bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(dataset[train_samples:], batch_size=self.bs, shuffle=False)
        return train_loader, val_loader
    
    def ClientUpdate(self):
        self.model.train()
        epoch_loss = []
  
        for iter in range(self.epochs):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                self.model.zero_grad()
                log_probs = self.model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        self.model.eval()
        return {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}, (sum(epoch_loss) / len(epoch_loss)) if epoch_loss else 0

    def train(self, n_iter):
        
        print(f"\nClient {self.client_id} starting training for {n_iter} iterations")
        
        for iter_idx in tqdm(range(n_iter), desc=f"Client {self.client_id}", leave=True):
            wk_cpu, loss = self.ClientUpdate()
            a = self.get_server_age()
            bl, bu = self.get_bounds()

            if a - self.ak > bu:
                print(f"Client {self.client_id}: Age gap too large ({a-self.ak} > {bu}), refreshing model")                
                w_cpu = self.get_server_model()
                w = {k: v.to(self.device) for k, v in w_cpu.items()}
                self.model.load_state_dict(w)
                self.ak = a
            elif  a-self.ak < bl:
                print(f"Client {self.client_id}: Age gap too small ({a-self.ak} < {bl}), skipping update")                
                continue

            else:
                print(f"Client {self.client_id}: Updating server (client age: {self.ak}, server age: {a})")                
                new_w_cpu, new_a = self.update_server(wk_cpu, self.ak, self.client_id)
                new_w = {k: v.to(self.device) for k, v in new_w_cpu.items()}
                self.model.load_state_dict(new_w)
                self.ak = new_a
        
        print(f"Client {self.client_id} completed training with final loss: {loss:.4f}")
        return self.client_id, loss

class Server:
    
    def __init__(self, model, dataset, bl, bu, n_clients, sample, n_iter=30):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Server initialized using {self.device}")
        self.bl = bl
        self.bu = bu
        
        print(f"Age bounds set to bl={bl}, bu={bu}")

        self.model = model.to(self.device)
        _initial_model = self.model().cpu()
        self.model_state_dict_cpu = _initial_model.state_dict()
        
        self.n_clients = n_clients
        self.data_partition = sample(dataset, n_clients)

        self.clients =  [Client(deepcopy(self.model_state_dict_cpu), self.data_partition[i], i, self.get_model_age, self.get_bounds, self.get_model, self.UpdateServer) for i in range(n_clients)]  
        self.a = bl

        self.lock = threading.Lock()
        self.n_iter = n_iter
        print(f"Server setup complete. Ready for training with {n_iter} iterations per client")    
    
    def get_bounds(self):
        return self.bl, self.bu 

    def get_model_age(self):
        return self.a

    def get_model(self):
        
        with self.lock:
            return deepcopy(self.model_state_dict_cpu)
    
    def UpdateServer(self, wk_cpu, ak):

        with self.lock:        

            alpha = math.pow(max(float(self.a - ak + 1), 1e-6), -0.5)
            alpha = max(0.0, min(1.0, alpha))

            # Aggregate models (on CPU)
            w_server_cpu = self.model_state_dict_cpu 
            new_sd_cpu = {}
            for key in w_server_cpu.keys():
                if key in wk_cpu:
                    server_tensor = w_server_cpu[key].float()
                    client_tensor = wk_cpu[key].float() 
                    new_sd_cpu[key] = (1.0 - alpha) * server_tensor + alpha * client_tensor
                else:
                    new_sd_cpu[key] = w_server_cpu[key] 

            self.model_state_dict_cpu = new_sd_cpu
            self.a += 1
            
            return deepcopy(new_sd_cpu), self.a

    async def train(self):

        print("\n=== Starting Federated Training ===")
        print(f"Number of clients: {self.n_clients}")
        print(f"Iterations per client: {self.n_iter}")


        threads = []
        client_results = {} 

        def run_thread(client_id):
            res = self.clients[client_id].train(self.n_iter)
            client_results[client_id] = res
        
        tic = time.time()

        for i in range(self.n_clients):
            thread  = threading.Thread(target=run_thread, args=(i, ))
            threads.append(thread)
            thread.start()            
            print(f"Client {i} thread started")
        
        with tqdm(total=len(threads), desc="Clients completed") as pbar:
            for thr in threads:
                thr.join()
                pbar.update(1)
                
        print("\n=== Client Results ===")        
        for client_id, result in client_results.items():
             if result:
                 final_loss = result
                 print(f"Client {client_id} finished with final avg loss: {final_loss:.4f}")
             else:
                 print(f"Client {client_id} encountered an error.")
        
        toc = time.time()
        print(f"\n--- Federated Training Completed ---")
        print(f"Final Server Age: {self.a}")
        print(f"Total Training Time: {toc-tic:.2f} seconds")
        
        return self.model_state_dict_cpu


        


