import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from copy import deepcopy

class FedSVRGClient:
    def __init__(self, model, dataset, client_id):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.dataset = dataset
        self.client_id = client_id

    def set_global_weights(self, global_model_state):
        self.model.load_state_dict(global_model_state)        

    def compute_gradients(self):
        
        self.model.train()

        init_grads = [torch.zeros_like(param) for param in self.model.parameters()] 

        n_samples = 0
        loader  = DataLoader(self.dataset, batch_size=64, shuffle=False)
        
        for i, (x, y) in enumerate(loader):
            x,y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            loss = F.cross_entropy(out, y)

            loss.backward()

            for i, param in enumerate(self.model.parameters()):
                if param.grad is not None:
                    init_grads[i] += param.grad * len(x)
            
            n_samples += len(x)

        for i in range(len(init_grads)):
            init_grads[i] /= n_samples
            
        return init_grads

    def local_update(self, full_grads, lr):
        
        self.model.train()
        
        global_model = deepcopy(self.model)
        global_model.load_state_dict(self.model.state_dict())
        global_model.train()

        loader = DataLoader(self.dataset, batch_size=64)

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device) 
            
            self.model.zero_grad()            
            out = self.model(x) 
            loss = F.cross_entropy(out, y)
            loss.backward()
            local_grad = self._get_gradients(self.model)

            global_model.zero_grad()            
            out = global_model(x) 
            loss = F.cross_entropy(out, y)
            loss.backward()
            gloabl_grad = self._get_gradients(global_model)

            reduced_grad = []

            for lg, gsg, fg in zip(local_grad, gloabl_grad, full_grads):
                reduced_grad.append(lg-gsg+fg)
            
            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), reduced_grad):
                    param.data.sub_(lr*grad)

        return self.model.state_dict()


    def _get_gradients(self, model):
        return [p.grad.clone() for p in model.parameters()]



class FedSVRGServer:
    def __init__(self, model, sample, args):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.n_clients = args.n_clienst

        self.data_partition = sample(args)

        self.clients =  [FedSVRGClient(deepcopy(model), self.data_partition[i], i) for i in range(self.n_clients)]  

        self.epochs = 10      
        self.lr = 1e-2

    def compute_full_gradient(self):

        full_grads = [torch.zeros_like(param) for param in self.model.parameters()]
        
        for client in self.clients:
            client.set_global_weights(self.model.state_dict())
            c_grad = client.compute_gradients()
            for i in range(len(full_grads)):
                full_grads[i] += c_grad[i]

        return [p/self.n_clients for p in full_grads]            

    def train(self):
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            full_gradients = self.compute_full_gradient()
            
            client_models = []
            
            for i, client in enumerate(self.clients):
                client.set_global_weights(self.model.state_dict())
                
                lr = self.lr / len(self.data_partition[i])
                client_model = client.local_update(full_gradients, lr)
                client_models.append(client_model)
            
            self._aggregate_models(client_models)
            
            if hasattr(self, 'evaluate'):
                accuracy = self.evaluate()
                print(f"Epoch {epoch+1} - Global model accuracy: {accuracy:.4f}")
        
        print("\nTraining completed!")
        return self.model.state_dict()
    
    def _aggregate_models(self, client_models):
        
        global_model_dict = self.model.state_dict()
        total_samples = sum(len(client.dataset) for client in self.clients)

        for key in global_model_dict.keys():
            global_model_dict[key] = torch.zeros_like(global_model_dict[key])
        
        for client_idx, client_model_dict in enumerate(client_models):
            weight = len(self.clients[client_idx].dataset) / total_samples
            for key in global_model_dict.keys():
                global_model_dict[key] += client_model_dict[key] * weight
        
        self.model.load_state_dict(global_model_dict)
