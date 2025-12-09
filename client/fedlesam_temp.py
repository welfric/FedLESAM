import torch
import torch.nn as nn 
import torch.nn.functional as F
from .client import Client
from utils import * 
from optimizer import *

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
    
class fedlesam_temp(Client):
    def __init__(self, device, model_func, received_vecs, dataset, lr, args): 
        super(fedlesam_temp, self).__init__(device, model_func, received_vecs, dataset, lr, args)
        
        self.tempnet = TempNet().to(self.device)
        self.temp_opt = torch.optim.SGD(
            self.tempnet.parameters(), 
            lr=lr
        )
        
        self.base_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=self.args.weight_decay)
        # Pass TempNet to LESAM
        self.optimizer = LESAMTemp(
            self.model.parameters(), 
            self.base_optimizer, 
            rho=self.args.rho,
            tempnet=self.tempnet,
            temp_opt=self.temp_opt
        )
    
    
    def train(self):
        # local training
        self.model.train()
        self.tempnet.train() # Set TempNet to train mode
        
        if self.received_vecs['global_update'] !=None:
            self.received_vecs['global_update'].to(self.device)
        
        # Initial zero_grad for TempNet optimizer
        self.temp_opt.zero_grad() 
        
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                
                # Pass all required objects to the LESAM optimizer
                self.optimizer.paras = [inputs, labels, self.loss, self.model]

                # --- NEW: Call step to perform SAM on both model and TempNet ---
                self.optimizer.step(self.received_vecs['global_update'])
                
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm) 
                
                # --- NEW: Step TempNet optimizer ---
                self.base_optimizer.step()
                self.temp_opt.step() # Update TempNet weights

        last_state_params_list = get_mdl_params(self.model)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        self.comm_vecs['local_model_param_list'] = last_state_params_list

        return self.comm_vecs