import torch
import torch.nn.functional as F
from utils import *  # Assuming get_mdl_params and set_client_from_params are here
from dataset import Dataset  # Keep if needed for type hints, otherwise remove
from torch.utils import data
from tqdm import tqdm
from time import time
from .client import Client


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


# FedAvg + TempNet Client Implementation
class fedavgt(Client):

    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        # Initialize the base client (sets up the core model, device, data, etc.)
        super().__init__(device, model_func, received_vecs, dataset, lr, args)

        # 1. TempNet Setup
        # Create and move TempNet to the device
        self.tempnet = TempNet().to(self.device)

        # 2. TempNet Optimizer
        # The core model's optimizer (self.optimizer) is kept for descent step
        # Create a separate optimizer for TempNet
        self.temp_opt = torch.optim.SGD(self.tempnet.parameters(), lr=lr)

    def train(self):
        """
        Overrides the base client's train method to use FedAvg with TempNet.
        Standard SGD with temperature scaling from TempNet.
        """
        self.model.train()
        self.tempnet.train()

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

                # Compute loss
                loss = self.loss(scaled_logits, labels)

                # Backward pass
                loss.backward()

                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.max_norm
                )

                # Update both model and tempnet
                self.optimizer.step()
                self.temp_opt.step()

        # --- Communication Vector Preparation (Same as base class) ---
        # Get the final model parameters as a vector
        last_state_params_list = get_mdl_params(self.model)

        # Compute update vector: Delta = w(i,K,t) - w(t)
        self.comm_vecs["local_update_list"] = (
            last_state_params_list - self.received_vecs["Params_list"]
        )
        # Store local model vector for the next round's RI or logging
        self.comm_vecs["local_model_param_list"] = last_state_params_list

        # Return the communication vector (only contains CNN model updates)
        return self.comm_vecs
