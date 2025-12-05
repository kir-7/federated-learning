import flwr as fl
import torch
from torch.utils.data import DataLoader

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train_dataset, val_dataset, config):
        self.cid = cid
        self.net = net
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=config.client_bs, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config.client_bs, shuffle=False)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def get_lr(self, round_num):
        # reduce lr by 0.9 every config.reduce_lr_every rounds
        return self.config.client_lr * (0.9**(round_num//self.config.reduce_lr_every))

    def fit(self, parameters, config):
        # should return parameters, num_examples, metrics 
        self.set_parameters(parameters)

        global_params = [torch.tensor(p).to(self.device) for p in parameters]
        self.net.train()

        lr = self.get_lr(config['server_round'])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        epoch_loss = []
        for _ in range(self.config.client_epochs):
            batch_loss = []
            for sample in self.train_loader:
                images, labels = sample['img'], sample['label']
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.net(images)

                # No proximal loss for now
                # proximal_term = 0.0
                # for local_weights, global_weights in zip(self.net.parameters(), global_params):
                #     proximal_term += (local_weights - global_weights).norm(2)**2
                # loss = criterion(output, labels) + (self.config.prox_lambda / 2) * proximal_term

                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # Return updated weights and metrics
        final_loss = sum(epoch_loss) / len(epoch_loss)
        return self.get_parameters(config={}), len(self.train_dataset), {"train_loss": final_loss, "data_samples":len(self.train_dataset)}

    def evaluate(self, parameters, config):
        # should return loss, num_examples used for evaluation and metrics
        self.set_parameters(parameters)
        self.net.eval()
        correct, total = 0, 0
        criterion = torch.nn.CrossEntropyLoss()
        losses = []
        with torch.no_grad():
            for sample in self.train_loader:
                images, labels = sample['img'], sample['label']
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)

                loss = criterion(outputs, labels)
                losses.append(loss.item())

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        final_loss = sum(losses) / len(losses)
        return final_loss, len(self.val_dataset), {"val_acc": accuracy, "data_samples":len(self.val_dataset), "val_loss":final_loss}