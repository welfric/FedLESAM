import torch
from client import *
from .server import Server


class FedSMOOT(Server):
    def __init__(
        self, device, model_func, init_model, init_par_list, datasets, method, args
    ):
        super(FedSMOOT, self).__init__(
            device, model_func, init_model, init_par_list, datasets, method, args
        )

        # rebuild
        self.comm_vecs = {
            "Params_list": init_par_list.clone().detach(),
        }
        self.Client = fedsmoot

        # FedSMOO specific parameters
        self.dual_variable_list = torch.zeros(
            (args.total_client, init_par_list.shape[0])
        )
        self.global_s = torch.zeros_like(init_par_list)
        self.rho = args.rho

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

    def postprocess(self, client):
        # Update dual variables and global smoothing vector
        self.dual_variable_list[client] += self.received_vecs["local_update_list"]
        if hasattr(self, "_current_s_i"):
            self.global_s += self._current_s_i / self.args.total_client

    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedSMOO with TempNet (ServerOpt) with dual variable aggregation
        # Normalize global_s
        if torch.norm(self.global_s) > 0:
            self.global_s = self.rho * self.global_s / torch.norm(self.global_s)

        # Apply dual variable correction
        dual_correction = torch.mean(self.dual_variable_list[selected_clients], dim=0)

        # w(t+1) = w(t) + eta_g * Delta + dual_correction
        return (
            self.server_model_params_list
            + self.args.global_learning_rate * Averaged_update
            + dual_correction
        )
