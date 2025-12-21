import torch
import torch.nn.functional as F
from utils import *
from .client import Client
from time import time
import copy


# TempNet for adaptive temperature scaling
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


class fedgmtt(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedgmtt, self).__init__(
            device, model_func, received_vecs, dataset, lr, args
        )

        # TempNet Setup
        feature_dim = args.feature_dim if hasattr(args, "feature_dim") else 512
        hidden_dim = args.temp_hidden_dim if hasattr(args, "temp_hidden_dim") else 128
        tau_min = args.tau_min if hasattr(args, "tau_min") else 0.05
        tau_max = args.tau_max if hasattr(args, "tau_max") else 2.0
        
        self.tempnet = TempNet(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            tau_min=tau_min,
            tau_max=tau_max
        ).to(self.device)

        # TempNet Optimizer
        self.temp_opt = torch.optim.SGD(self.tempnet.parameters(), lr=lr)

        # FedGMT specific parameters
        self.EMA = None
        self.KLDiv = torch.nn.KLDivLoss(reduction="batchmean")
        self.tau = args.tau if hasattr(args, "tau") else 1.0
        self.gama = args.gama if hasattr(args, "gama") else 0.1
        self.beta = 1 / args.beta if hasattr(args, "beta") else 1.0
        self.dual_variable = None

    def train(self):
        # Local training with FedGMT + TempNet
        self.model.train()
        self.tempnet.train()

        # Disable gradients for EMA model
        if self.EMA is not None:
            for params in self.EMA.parameters():
                params.requires_grad = False

        # Store initial parameters
        with torch.no_grad():
            regular_params = param_to_vector(self.model).detach().clone()

        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()

                # Forward pass with EMA model (for knowledge distillation)
                with torch.no_grad():
                    output_ema = self.EMA(inputs) if self.EMA is not None else None

                # Forward pass with main model - extract features
                # Check if model supports return_feats
                if hasattr(self.model, 'return_feats') or 'return_feats' in str(type(self.model)):
                    try:
                        feats, logits = self.model(inputs, return_feats=True)
                    except:
                        # Fallback: use logits as features
                        logits = self.model(inputs)
                        feats = logits.detach()
                else:
                    logits = self.model(inputs)
                    feats = logits.detach()

                # Get temperature scaling from TempNet
                tau_temp = self.tempnet(feats.detach())

                # Scale logits by temperature
                scaled_logits = logits / tau_temp

                # Main task loss
                loss_main = self.loss(scaled_logits, labels)

                # Knowledge distillation loss (GMT loss) using EMA model
                if self.EMA is not None and output_ema is not None:
                    pred_probs = F.log_softmax(scaled_logits / self.tau, dim=1)
                    ema_probs = torch.softmax(output_ema / self.tau, dim=1)
                    loss_kl = (
                        self.gama * (self.tau**2) * self.KLDiv(pred_probs, ema_probs)
                    )
                    loss = loss_main + loss_kl
                else:
                    loss = loss_main

                # Dual variable correction (DYN-style)
                if self.dual_variable is not None:
                    local_params = param_to_vector(self.model)
                    # Ensure dual_variable is on the same device
                    dual_var = self.dual_variable.to(self.device)
                    loss -= torch.dot(local_params, (-self.beta) * dual_var)

                self.optimizer.zero_grad()
                self.temp_opt.zero_grad()
                loss.backward()

                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.max_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.tempnet.parameters(), max_norm=self.max_norm
                )

                self.optimizer.step()
                self.temp_opt.step()

        # Prepare communication vectors
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs["local_update_list"] = (
            last_state_params_list - self.received_vecs["Params_list"]
        )
        self.comm_vecs["local_model_param_list"] = last_state_params_list

        # Store local update for server synchronization
        with torch.no_grad():
            local_params = param_to_vector(self.model).detach()
            self.local_update = local_params - regular_params

        return self.comm_vecs