import torch
import copy
from utils import *
from client import *
from .server import Server


class FedGMT(Server):
    def __init__(
        self, device, model_func, init_model, init_par_list, datasets, method, args
    ):
        super(FedGMT, self).__init__(
            device, model_func, init_model, init_par_list, datasets, method, args
        )

        # rebuild
        self.comm_vecs = {
            "Params_list": init_par_list.clone().detach(),
        }
        self.Client = fedgmt

        # FedGMT specific parameters
        self.EMA_model = copy.deepcopy(init_model)
        self.dual_variable_list = torch.zeros(
            (args.total_client, init_par_list.shape[0])
        ).to(device)
        self.alpha = args.alpha if hasattr(args, "alpha") else 0.1

    def process_for_communication(self, client, Averaged_update):
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
        """Send EMA model to client before training"""
        _edge_device.EMA = copy.deepcopy(self.EMA_model)
        _edge_device.dual_variable = self.dual_variable_list[client_id].clone()

    def postprocess(self, client):
        # Update dual variables
        self.dual_variable_list[client] = self.dual_variable_list[
            client
        ] + self.received_vecs["local_update_list"].to(self.device)

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedGMT (ServerOpt) with EMA model update and dual variable aggregation
        # Apply dual variable correction
        dual_correction = torch.mean(
            self.dual_variable_list[selected_clients], dim=0
        ).to(self.device)

        # w(t+1) = w(t) + eta_g * Delta + dual_correction
        new_params = (
            self.server_model_params_list.to(self.device)
            + self.args.global_learning_rate * Averaged_update.to(self.device)
            + dual_correction
        )

        # Update EMA model: EMA = EMA * alpha + global * (1 - alpha)
        with torch.no_grad():
            ema_params = param_to_vector(self.EMA_model).to(self.device)
            ema_params = ema_params * self.alpha + new_params * (1 - self.alpha)
            set_client_from_params(self.device, self.EMA_model, ema_params)

        return new_params
