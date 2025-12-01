# template.py
PINN_TEMPLATE = """\
import math, torch, torch.nn as nn, torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
import os

# Hyperparameters
HIDDEN_LAYERS = {hidden_layers}
HIDDEN_WIDTH  = {hidden_width}
ACTIVATION    = "{activation}"
OPTIMIZER     = "{optimizer}"
LR            = {lr}
EPOCHS        = {epochs}
N_COL         = {pde_collocation}
BC_WEIGHT     = {bc_weight}

# Scheduler params
LR_SCHEDULER_TYPE = "{lr_scheduler_type}"
LR_DECAY_GAMMA    = {lr_decay_gamma}

# Problem Constants
IN_DIM = {input_dim}
OUT_DIM = {output_dim}
DOMAIN_MIN = {domain_min}
DOMAIN_MAX = {domain_max}

def _act(name):
    return dict(tanh=nn.Tanh, relu=nn.ReLU, gelu=nn.GELU, silu=nn.SiLU, sin=torch.sin)[name]

class PINN(nn.Module):
    def __init__(self, in_dim=IN_DIM, out_dim=OUT_DIM):
        super().__init__()
        self.min_val = DOMAIN_MIN
        self.max_val = DOMAIN_MAX
        layers = [nn.Linear(in_dim, HIDDEN_WIDTH)]
        if ACTIVATION == "sin":
             pass
        else:
             layers.append(_act(ACTIVATION) if not isinstance(_act(ACTIVATION), type) else _act(ACTIVATION)())

        for _ in range(HIDDEN_LAYERS-1):
            layers.append(nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH))
            if ACTIVATION != "sin":
                layers.append(_act(ACTIVATION) if not isinstance(_act(ACTIVATION), type) else _act(ACTIVATION)())
        
        layers.append(nn.Linear(HIDDEN_WIDTH, out_dim))
        self.net = nn.Sequential(*layers)
        
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.max_val - self.min_val > 1e-6:
            x_norm = 2.0 * (x - self.min_val) / (self.max_val - self.min_val) - 1.0
        else:
            x_norm = x

        if ACTIVATION == "sin":
            for layer in self.net:
                x_norm = layer(x_norm)
                if isinstance(layer, nn.Linear) and layer != self.net[-1]:
                    x_norm = torch.sin(x_norm)
            return x_norm
        else:
            return self.net(x_norm)

def physics_loss(model, x_fixed):
    x = x_fixed.detach().requires_grad_(True)
    y = model(x)
    {physics_code}

def boundary_loss(model, device):
    {boundary_code}

def train_and_evaluate(seed=42, round_num=0, plot_filename="pinn_result.png"):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN().to(device)
    
    x_fixed_raw = torch.rand(N_COL, IN_DIM, device=device)
    x_fixed_val = (DOMAIN_MAX - DOMAIN_MIN) * x_fixed_raw + DOMAIN_MIN
    x_fixed = x_fixed_val.detach()

    print(f"--- Round {round_num} Start Training (Opt: {OPTIMIZER}, Epochs: {EPOCHS}) ---")

    opt = None
    scheduler = None
    loss_history = [] 

    # 1. --- Phase 1: Adam Warmup
    if OPTIMIZER == "lbfgs":
        print(">>> Phase 1: Adam Warmup (2000 steps) to escape trivial solution...")
        opt_warmup = torch.optim.Adam(model.parameters(), lr=1e-3)
        for i in range(2000):
            opt_warmup.zero_grad()
            loss = physics_loss(model, x_fixed) + BC_WEIGHT * boundary_loss(model, device)
            loss.backward()
            opt_warmup.step()
        print(">>> Warmup finished. Switching to L-BFGS.")

    if OPTIMIZER == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        
        # Scheduler setup
        if hasattr(torch.optim.lr_scheduler, "StepLR") and "{lr_scheduler_type}" == "step":
            step_size = max(1, int(EPOCHS / 3))
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma={lr_decay_gamma})
        elif hasattr(torch.optim.lr_scheduler, "CosineAnnealingLR") and "{lr_scheduler_type}" == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
            
        for epoch in range(EPOCHS):
            opt.zero_grad()
            loss = physics_loss(model, x_fixed) + BC_WEIGHT * boundary_loss(model, device)
            loss.backward()
            opt.step()
            loss_history.append(loss.item())
            
            if scheduler:
                scheduler.step()
            
            if epoch % 1000 == 0 or epoch == EPOCHS - 1:
                print(f"[Adam] Epoch {epoch:5d}/{EPOCHS} | Loss: {loss.item():.6e}")
            
    elif OPTIMIZER == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=EPOCHS, max_eval=EPOCHS*1.25, 
                                history_size=50,line_search_fn="strong_wolfe",tolerance_grad=1e-7, tolerance_change=1e-9)
        lbfgs_iter = 0
        def closure():
            nonlocal lbfgs_iter
            opt.zero_grad()
            loss = physics_loss(model, x_fixed) + BC_WEIGHT * boundary_loss(model, device)
            loss.backward()
            loss_history.append(loss.item())
            
            if lbfgs_iter % 1000 == 0:
                print(f"[LBFGS] Iter {lbfgs_iter:5d} | Loss: {loss.item():.6e}")
            lbfgs_iter += 1
            return loss
        opt.step(closure)

    # --- 1 ---
    mse = -1.0 
    mae = -1.0
    with torch.no_grad():
        if IN_DIM == 1:
            xs_plot = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, 200, device=device).unsqueeze(1)
            pred_plot = model(xs_plot)
        else:
            xs_plot = None
            pred_plot = None
        
        plt.figure(figsize=(8, 5))
        # --- INJECTED PLOTTING CODE START ---
        {plotting_code}
        # --- INJECTED PLOTTING CODE END ---
        plt.legend()
        plt.title(f"Round {round_num} Prediction (MSE: {mse:.2e})")
        plt.grid(True)
        pred_image_path = f"pinn_pred_round_{round_num}_seed_{seed}.png"
        plt.savefig(pred_image_path)
        plt.close()

    # --- 2 ---
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Total Loss', color='purple', linewidth=1.5)
    plt.yscale('log') 
    plt.xlabel('Iterations')
    plt.ylabel('Loss (Log Scale)')
    plt.title(f"Round {round_num} Loss History")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend()
    
    loss_image_path = f"pinn_loss_round_{round_num}_seed_{seed}.png"
    plt.savefig(loss_image_path)
    plt.close()

    final_loss = physics_loss(model, x_fixed) + BC_WEIGHT * boundary_loss(model, device)
    print(f"--- Round {round_num} Finished. Final Loss: {final_loss.item():.6e} ---\\n")
        
    return {
        "final_loss": final_loss.item(), 
        "mse": mse, 
        "mae": mae, 
        "image_path": pred_image_path, 
        "loss_image_path": loss_image_path
    }
"""