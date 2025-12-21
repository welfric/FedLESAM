import torch
import copy
from utils import *
from client import *
from .server import Server


class FedGMTT(Server):
    def __init__(
        self, device, model_func, init_model, init_par_list, datasets, method, args
    ):
        super(FedGMTT, self).__init__(
            device, model_func, init_model, init_par_list, datasets, method, args
        )

        # Rebuild communication vectors
        self.comm_vecs = {
            "Params_list": init_par_list.clone().detach(),
        }
        
        # Import the client class with TempNet
        self.Client = fedgmtt

        # FedGMT specific parameters
        self.EMA_model = copy.deepcopy(init_model).to(device)
        self.dual_variable_list = torch.zeros(
            (args.total_client, init_par_list.shape[0]),
            device=device  # Initialize directly on the correct device
        )
        self.alpha = args.alpha if hasattr(args, "alpha") else 0.1

    def process_for_communication(self, client, Averaged_update):
        """Prepare parameters to send to client"""
        if not self.args.use_RI:
            self.comm_vecs["Params_list"].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs["Params_list"].copy_(
                self.server_model_params_list
                + self.args.beta
                * (self.server_model_params_list - self.clients_params_list[client])
            )

    def send_ema_to_client(self, _edge_device, client_id):
        """Send EMA model and dual variable to client before training"""
        _edge_device.EMA = copy.deepcopy(self.EMA_model).to(_edge_device.device)
        _edge_device.dual_variable = self.dual_variable_list[client_id].clone().detach()

    def postprocess(self, client):
        """Update dual variables after receiving client updates"""
        # Ensure tensors are on the same device
        local_update = self.received_vecs["local_update_list"].to(self.device)
        self.dual_variable_list[client] = (
            self.dual_variable_list[client] + local_update
        )

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        """FedGMTT global update with EMA model and dual variable aggregation"""
        # Ensure all tensors are on the correct device
        Averaged_update = Averaged_update.to(self.device)
        
        # Apply dual variable correction (average over selected clients)
        dual_correction = torch.mean(
            self.dual_variable_list[selected_clients], dim=0
        )

        # w(t+1) = w(t) + eta_g * Delta + dual_correction
        new_params = (
            self.server_model_params_list
            + self.args.global_learning_rate * Averaged_update
            + dual_correction
        )

        # Update EMA model: EMA = EMA * alpha + global * (1 - alpha)
        with torch.no_grad():
            ema_params = param_to_vector(self.EMA_model)
            ema_params = ema_params * self.alpha + new_params * (1 - self.alpha)
            set_client_from_params(self.device, self.EMA_model, ema_params)

        return new_params