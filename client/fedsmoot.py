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


class fedsmoot(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedsmoot, self).__init__(
            device, model_func, received_vecs, dataset, lr, args
        )

        # 1. TempNet Setup
        self.tempnet = TempNet().to(self.device)

        # 2. TempNet Optimizer
        self.temp_opt = torch.optim.SGD(self.tempnet.parameters(), lr=lr)

        # FedSMOO specific parameters
        self.beta = 1 / args.beta
        self.global_s = None
        self.rho = args.rho
        self.mu_i = torch.zeros_like(param_to_vector(self.model).detach())
        self.dual_variable = None
        self.local_s_i = None

    def train(self):
        # local training with FedSMOO + TempNet
        self.model.train()
        self.tempnet.train()

        # Store initial parameters
        with torch.no_grad():
            regular_params = param_to_vector(self.model).detach().clone()

        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                # Forward pass with feature extraction
                feats, logits = self.model(inputs, return_feats=True)

                # Get temperature scaling from TempNet
                tau = self.tempnet(feats.detach())

                # Scale logits by temperature
                scaled_logits = logits / tau

                # Compute loss
                loss = self.loss(scaled_logits, labels)

                # Backward to get gradients
                self.optimizer.zero_grad()
                self.temp_opt.zero_grad()
                loss.backward()

                # Flatten gradients
                grad_batch = (
                    torch.nn.utils.parameters_to_vector(
                        [p.grad for p in self.model.parameters() if p.grad is not None]
                    )
                    .detach()
                    .clone()
                )

                # Apply SMOO correction: gradient = gradient - mu_i - global_s
                if self.global_s is not None:
                    grad_batch = grad_batch - self.mu_i - self.global_s

                # Assign corrected gradients back
                idx = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        numel = p.grad.numel()
                        p.grad.data = grad_batch[idx : idx + numel].view_as(p.grad)
                        idx += numel

                # Update mu_i and perform step
                self.optimizer.step()
                self.temp_opt.step()

                # Store s_i_k for dual variable update
                with torch.no_grad():
                    s_i_k = grad_batch.clone()
                    self.mu_i += (
                        (s_i_k - self.global_s) if self.global_s is not None else s_i_k
                    )

                # Apply DYN correction (proximal term)
                local_params = param_to_vector(self.model)
                loss_dyn = (self.beta / 2) * torch.norm(
                    local_params - regular_params, 2
                ) ** 2

                if self.dual_variable is not None:
                    loss_dyn -= torch.dot(
                        local_params, (-self.beta) * self.dual_variable
                    )

                self.optimizer.zero_grad()
                self.temp_opt.zero_grad()
                loss_dyn.backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.max_norm
                )
                self.optimizer.step()
                self.temp_opt.step()

        # Prepare communication vectors
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs["local_update_list"] = (
            last_state_params_list - self.received_vecs["Params_list"]
        )
        self.comm_vecs["local_model_param_list"] = last_state_params_list

        # Store for server synchronization
        with torch.no_grad():
            local_params = param_to_vector(self.model).detach()
            self.local_update = local_params - regular_params
            self.local_s_i = self.mu_i.clone()

        return self.comm_vecs
