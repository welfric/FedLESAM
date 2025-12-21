import torch
import torch.nn.functional as F
from utils import *
from .client import Client
from time import time


# Placeholder for TempNet (you should use your actual TempNet definition)
class TempNet(torch.nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128, tau_min=0.05, tau_max=2.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(feature_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        self.tau_min = tau_min
        self.tau_max = tau_max

    def forward(self, x):
        h = F.relu(self.fc1(x))
        raw = self.fc2(h)
        tau = torch.sigmoid(raw)
        tau = tau * (self.tau_max - self.tau_min) + self.tau_min
        return tau.mean()


class scaffoldt(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(scaffoldt, self).__init__(
            device, model_func, received_vecs, dataset, lr, args
        )

        # 1. TempNet Setup
        # Create and move TempNet to the device
        self.tempnet = TempNet().to(self.device)

        # 2. TempNet Optimizer
        # Create a separate optimizer for TempNet
        self.temp_opt = torch.optim.SGD(self.tempnet.parameters(), lr=lr)

    def train(self):
        # local training with variance reduction and TempNet
        self.model.train()
        self.tempnet.train()
        t_start = time()
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                # Zero gradients
                self.optimizer.zero_grad()
                self.temp_opt.zero_grad()

                # Forward pass with feature extraction
                feats, logits = self.model(inputs, return_feats=True)

                # Get temperature scaling from TempNet
                tau = self.tempnet(feats.detach())

                # Scale logits by temperature
                scaled_logits = logits / tau

                # Main loss
                loss_pred = self.loss(scaled_logits, labels)

                # SCAFFOLD variance reduction term
                param_list = param_to_vector(self.model)
                delta_list = self.received_vecs["Local_VR_correction"].to(self.device)
                loss_correct = torch.sum(param_list * delta_list)

                # Combined loss
                loss = loss_pred + loss_correct

                loss.backward()

                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.max_norm
                )

                # Update both model and tempnet
                self.optimizer.step()
                self.temp_opt.step()
        t_end = time()
        print(t_end - t_start)
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs["local_update_list"] = (
            last_state_params_list - self.received_vecs["Params_list"]
        )
        self.comm_vecs["local_model_param_list"] = last_state_params_list

        return self.comm_vecs
