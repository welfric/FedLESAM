import torch
import torch.nn.functional as F
from utils import *
from .client import Client
from time import time
import copy


class fedgmt(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args):
        super(fedgmt, self).__init__(
            device, model_func, received_vecs, dataset, lr, args
        )

        # FedGMT specific parameters
        self.EMA = None
        self.KLDiv = torch.nn.KLDivLoss(reduction="batchmean")
        self.tau = args.tau if hasattr(args, "tau") else 1.0
        self.gama = args.gama if hasattr(args, "gama") else 0.1
        self.beta = 1 / args.beta if hasattr(args, "beta") else 1.0
        self.dual_variable = None

    def train(self):
        # Local training with FedGMT
        self.model.train()

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

                # Forward pass with main model
                output = self.model(inputs)

                # Main task loss
                loss_main = self.loss(output, labels)

                # Knowledge distillation loss (GMT loss) using EMA model
                if self.EMA is not None and output_ema is not None:
                    pred_probs = F.log_softmax(output / self.tau, dim=1)
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
                loss.backward()

                # Clip gradients to prevent exploding
                torch.nn.utils.clip_grad_norm_(
                    parameters=self.model.parameters(), max_norm=self.max_norm
                )

                self.optimizer.step()

        # Prepare communication vectors
        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs["local_update_list"] = (
            last_state_params_list - self.received_vecs["Params_list"].to(self.device)
        )
        self.comm_vecs["local_model_param_list"] = last_state_params_list

        # Store local update for server synchronization
        with torch.no_grad():
            local_params = param_to_vector(self.model).detach()
            self.local_update = local_params - regular_params

        return self.comm_vecs