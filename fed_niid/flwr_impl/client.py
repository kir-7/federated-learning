import flwr as fl
import torch

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train_loader, val_loader, config):
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # 1. Update local model with global (or graph-averaged) parameters
        self.set_parameters(parameters)
        
        # 2. Save these parameters as the "Global/Anchor" state for FedProx
        global_params = [torch.tensor(p).to(self.device) for p in parameters]
        
        # 3. Training Loop
        self.net.train()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.client_lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        epoch_loss = []
        for _ in range(self.config.client_epochs):
            batch_loss = []
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.net(images)
                
                # --- FedProx Loss ---
                proximal_term = 0.0
                for local_weights, global_weights in zip(self.net.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                
                loss = criterion(output, labels) + (self.config.prox_lambda / 2) * proximal_term
                # --------------------

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # Return updated weights and metrics
        final_loss = sum(epoch_loss) / len(epoch_loss)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"train_loss": final_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.net.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return 0.0, len(self.val_loader.dataset), {"val_acc": accuracy}