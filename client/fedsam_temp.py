
import torch
import torch.nn.functional as F
from utils import * # Assuming get_mdl_params and set_client_from_params are here
from dataset import Dataset # Keep if needed for type hints, otherwise remove
from torch.utils import data
from tqdm import tqdm
from time import time
from client import *

# Assume TempNet and SimpleCNN models are accessible (e.g., imported from models)
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

# Your Base Client Class (Included for context)
# class Client():
#     ... (The code you provided) ...

# ----------------------------------------------------------------------
# FedSAM + TempNet Client Implementation
# ----------------------------------------------------------------------

class fedsam_temp(Client):
    
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
        
        # 3. SAM Perturbation Radius (rho should be in args, but set a default)
        self.rho = getattr(self.args, 'rho', 0.05) 
        
        # The base client's self.loss (CrossEntropyLoss) will be used.
        
    @torch.no_grad()
    def _compute_perturbation_and_ascent(self, inputs, labels):
        """
        Executes the first SAM step: 
        1. Compute loss and gradient at w.
        2. Calculate perturbation (epsilon).
        3. Save w, ascend to w + epsilon.
        """
        self.model.train()
        self.tempnet.train()
        
        # Zero gradients for both model and tempnet
        self.optimizer.zero_grad() 
        self.temp_opt.zero_grad()

        # Step 1: Forward at w and compute gradient
        feats, logits = self.model(inputs)
        # Detach features when computing loss for the first gradient to prevent 
        # TempNet from influencing the first-step gradient calculation for SAM.
        tau = self.tempnet(feats.detach()) 
        scaled_logits = logits / tau
        loss = self.loss(scaled_logits, labels)
        loss.backward()

        # Calculate gradient norm (Model Only)
        grad_norm = torch.nn.utils.parameters_to_vector(
            [p.grad for p in self.model.parameters() if p.grad is not None]
        ).norm()

        if grad_norm == 0:
            return None # Skip if gradient is zero

        # SAVE ORIGINAL MODEL WEIGHTS
        original_params = [p.clone() for p in self.model.parameters()]
        
        # Compute perturbation ε = ρ * ∇L / ||∇L||
        eps = self.rho * torch.nn.utils.parameters_to_vector(
            [p.grad / (grad_norm + 1e-12) for p in self.model.parameters() if p.grad is not None]
        )

        # Ascent step: w ← w + ε (MODEL ONLY)
        idx = 0
        for p in self.model.parameters():
            if p.grad is not None:
                numel = p.grad.numel()
                p.data.add_(eps[idx:idx+numel].view_as(p))
                idx += numel
                
        return original_params

    def _second_step_and_descent(self, inputs, labels, original_params):
        """
        Executes the second SAM step: 
        1. Compute loss and gradient at w + epsilon.
        2. Restore original w.
        3. Perform descent step using gradient calculated at w + epsilon.
        """
        # --- Step 2: Compute gradient at perturbed point (w + ε) ---
        self.optimizer.zero_grad() 
        self.temp_opt.zero_grad()
        
        # Forward pass on perturbed model
        feats_pert, logits_pert = self.model(inputs)
        
        # Calculate loss for second gradient
        # Note: We now allow the gradient to flow back through TempNet if needed,
        # but the standard FedSAM+Temp paper often optimizes TempNet only on the
        # perturbed loss, and it doesn't need to contribute to the SAM ascent/descent.
        # We use feats_pert.detach() for the TempNet for simplicity and stability,
        # ensuring TempNet is trained to predict tau based on the features 
        # from the model at w+epsilon.
        tau_pert = self.tempnet(feats_pert.detach()) 
        loss_pert = self.loss(logits_pert / tau_pert, labels)
        
        # Compute gradients for both model and tempnet
        loss_pert.backward() 

        # RESTORE ORIGINAL MODEL WEIGHTS
        with torch.no_grad():
            for p, p_orig in zip(self.model.parameters(), original_params):
                p.data.copy_(p_orig)

        # Descent step using gradients calculated at w+eps
        # self.optimizer.step() applies the gradient to the restored weights (w)
        self.optimizer.step()
        self.temp_opt.step() # Update TempNet using its gradient calculated from loss_pert
        
        # Clip gradients (optional, but in the base client)
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_norm)


    def train(self):
        """
        Overrides the base client's train method to use FedSAM + TempNet updates.
        """
        self.model.train()
        self.tempnet.train()
        
        for k in range(self.args.local_epochs):
            for i, (inputs, labels) in enumerate(self.dataset):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).reshape(-1).long()
                
                # FedSAM + TempNet two-step update
                original_params = self._compute_perturbation_and_ascent(inputs, labels)
                if original_params is None: # Skip if grad_norm was zero
                    continue 

                self._second_step_and_descent(inputs, labels, original_params)
                
        # --- Communication Vector Preparation (Same as base class) ---
        # Get the final model parameters as a vector
        last_state_params_list = get_mdl_params(self.model) 
        
        # Compute update vector: Delta = w(i,K,t) - w(t)
        self.comm_vecs['local_update_list'] = last_state_params_list - self.received_vecs['Params_list']
        # Store local model vector for the next round's RI or logging
        self.comm_vecs['local_model_param_list'] = last_state_params_list 

        # Return the communication vector (only contains CNN model updates)
        return self.comm_vecs