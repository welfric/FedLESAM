import torch
import torch.nn.functional as F

class LESAMTemp(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, tempnet, temp_opt, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LESAMTemp, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        
        self.tempnet = tempnet
        self.temp_opt = temp_opt
        
        for group in self.param_groups:
            group["rho"] = rho
        self.paras = None
        
    # first_step and second_step remain the same, as they only operate 
    # on the self.param_groups (main model weights)
    @torch.no_grad()
    def first_step(self,g_update):
        # ... (same as original: calculates grad_norm from g_update, 
        # calculates e_w, and performs p.add_(e_w)) ...
        grad_norm = 0
        for group in self.param_groups:
            for idx,p in enumerate(group["params"]):
                p.requires_grad = True 
                if g_update ==None: 
                    continue
                else:
                    # Note: You were calculating the norm of g_update for the scale here.
                    # This implementation is specific to FedLESAM's definition of global update.
                    grad_norm+=g_update[idx].norm(p=2) 
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx,p in enumerate(group["params"]):
                p.requires_grad = True 
                if g_update ==None: 
                    continue
                else:
                    e_w=-g_update[idx] * scale.to(p)
                p.add_(e_w * 1)
                self.state[p]["e_w"] = e_w


    @torch.no_grad()
    def second_step(self):
        # ... (same as original: performs p.sub_(self.state[p]["e_w"]) to restore w) ...
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                p.sub_(self.state[p]["e_w"]) 
                self.state[p]["e_w"] = 0


    def step(self,g_update=None):
        inputs, labels, loss_func, model = self.paras
        
        # 1. SAM Ascent (Perturb w to w + e(w))
        self.first_step(g_update)

        # 2. Forward Pass at w + e(w) with TempNet
        # model(inputs) must now return (feats, logits)
        feats, logits = model(inputs, return_feats=True) # Assuming model supports this
        
        # Detach features for TempNet prediction in the SAM ascent step 
        # (TempNet is trained in the descent step)
        tau = self.tempnet(feats.detach())
        scaled_predictions = logits / tau
        
        # 3. Backward Pass at w + e(w)
        loss = loss_func(scaled_predictions, labels)
        
        # Clear gradients for both optimizers
        self.zero_grad() # Clears main model gradients
        self.temp_opt.zero_grad() # Clears TempNet gradients
        
        loss.backward()

        # 4. SAM Descent (Restore w from w + e(w))
        self.second_step()
        
        # 5. Backward Pass at w with TempNet
        # Since TempNet's update is part of the overall descent, 
        # the loss.backward() above calculates the gradient for TempNet as well.
        # TempNet's gradient (dLoss/dTempNet_weights) flows through tau.
        
        # The base_optimizer.step() and temp_opt.step() are called in fedlesam.train().