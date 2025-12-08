import flwr as fl
import torch
from torch.utils.data import DataLoader, Subset

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, train_dataset, val_dataset, config):
        self.cid = cid
        self.net = net
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # datasets are based on labels

        # sort the dataset to introduce new classes on each data drift
        self.train_dataset_sorted = train_dataset
        self.val_dataset_sorted = val_dataset

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.n_train_samples = len(self.train_dataset)
        self.n_val_samples = len(self.val_dataset)

        self.init_data_usage : float = config.init_data_usage
        self.total_drift_steps = max(1, config.global_rounds // config.data_drift_every)

        if self.total_drift_steps > 1:
            self.growth_per_step = (1.0 - self.init_data_usage) / (self.total_drift_steps - 1)
        else:
            self.growth_per_step = 0.0 # If only 1 step, we stick to init or jump to 1.0 depending on preference

        self.train_slice_idx = 0
        self.val_slice_idx = 0

    def _get_current_pct(self, server_round):
        current_step = (server_round - 1) // self.config.data_drift_every

        current_pct = self.init_data_usage + (current_step * self.growth_per_step)

        return min(current_pct, 1.0)

    def get_round_train_data(self, server_round : int):

        current_pct = self._get_current_pct(server_round)

        self.train_slice_idx = max(int(self.n_train_samples * current_pct), 1)

        print(f"[Round {server_round}] Client using {self.train_slice_idx}/{self.n_train_samples} train samples ({int(current_pct*100)}%)")

        return Subset(self.train_dataset_sorted, list(range(self.train_slice_idx)))

    def get_round_val_data(self, server_round : int):
        current_pct = self._get_current_pct(server_round)

        self.val_slice_idx = max(int(self.n_val_samples * current_pct), 1)

        print(f"[Round {server_round}] Client using {self.val_slice_idx}/{self.n_val_samples} val samples ({int(current_pct*100)}%)")

        return Subset(self.val_dataset_sorted, list(range(self.val_slice_idx)))

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

        train_data = self.get_round_train_data(config['server_round'])

        train_loader = DataLoader(train_data, batch_size=min(self.config.client_bs, self.train_slice_idx), shuffle=True)

        global_params = [torch.tensor(p).to(self.device) for p in parameters]
        self.net.train()

        lr = self.get_lr(config['server_round'])
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        epoch_loss = []
        for _ in range(self.config.client_epochs):
            batch_loss = []
            for sample in train_loader:
                images, labels = sample['img'], sample['label']
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.net(images)

                proximal_term = 0.0
                for local_weights, global_weights in zip(self.net.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)**2
                loss = criterion(output, labels) + (self.config.prox_lambda / 2) * proximal_term

                # loss = criterion(output, labels)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # Return updated weights and metrics
        final_loss = sum(epoch_loss) / len(epoch_loss)
        return self.get_parameters(config={}), len(train_data), {"train_loss": final_loss, "data_samples":len(train_data)}

    def evaluate(self, parameters, config):
        # should return loss, num_examples used for evaluation and metrics
        self.set_parameters(parameters)

        val_data = self.get_round_val_data(config["server_round"])
        val_loader = DataLoader(val_data, batch_size=min(self.config.client_bs, self.val_slice_idx))

        self.net.eval()
        correct, total = 0, 0
        criterion = torch.nn.CrossEntropyLoss()
        losses = []
        with torch.no_grad():
            for sample in val_loader:
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
        return final_loss, len(val_data), {"val_acc": accuracy, "data_samples":len(val_data), "val_loss":final_loss}