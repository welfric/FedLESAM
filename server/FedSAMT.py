from .server import Server
from client import fedsam_temp
import torch

class FedSAMTemp(Server): # Renamed for clarity
    def __init__(self, device, model_func, init_model, init_par_list, datasets, method, args): 
        super(FedSAMTemp, self).__init__(device, model_func, init_model, init_par_list, datasets, method, args)
        
        self.comm_vecs = {
            'Params_list': init_par_list.clone().detach(),
        }
        # Use the newly defined FedSAM + Temp client
        self.Client = fedsam_temp 
    
    
    def process_for_communication(self, client, Averaged_update):
        # Implementation for RI or simple parameter broadcast
        if not self.args.use_RI:
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list)
        else:
            # RI adopts the w(i,t) = w(t) + beta[w(t) - w(i,K,t-1)] as initialization
            self.comm_vecs['Params_list'].copy_(self.server_model_params_list + self.args.beta\
                                 * (self.server_model_params_list - self.clients_params_list[client]))
        
        
    def global_update(self, selected_clients, Averaged_update, Averaged_model):
        # FedSAM Server: w(t+1) = w(t) + eta_g * Delta
        return self.server_model_params_list + self.args.global_learning_rate * Averaged_update