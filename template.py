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
        # 第一层后加激活
        if ACTIVATION == "sin":
             pass # sin 不需要实例化 nn.Module，直接在 forward 用 torch.sin
        else:
             layers.append(_act(ACTIVATION) if not isinstance(_act(ACTIVATION), type) else _act(ACTIVATION)())

        for _ in range(HIDDEN_LAYERS-1):
            layers.append(nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH))
            if ACTIVATION != "sin":
                layers.append(_act(ACTIVATION) if not isinstance(_act(ACTIVATION), type) else _act(ACTIVATION)())
        
        layers.append(nn.Linear(HIDDEN_WIDTH, out_dim))
        self.net = nn.Sequential(*layers)
        
        # 初始化
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 归一化输入到 [-1, 1]
        if self.max_val - self.min_val > 1e-6:
            x_norm = 2.0 * (x - self.min_val) / (self.max_val - self.min_val) - 1.0
        else:
            x_norm = x
            
        # 处理 sin 激活 (因为 torch.sin 不是 nn.Module)
        if ACTIVATION == "sin":
            for layer in self.net:
                x_norm = layer(x_norm)
                if isinstance(layer, nn.Linear) and layer != self.net[-1]:
                    x_norm = torch.sin(x_norm)
            return x_norm
        else:
            return self.net(x_norm)

# 【关键修改】把采样点作为参数传入，而不是在函数内生成
def physics_loss(model, x_fixed):
    # 直接使用传入的固定点
    x = x_fixed.detach().requires_grad_(True)
    y = model(x)
    # --- INJECTED PHYSICS CODE START ---
    {physics_code}
    # --- INJECTED PHYSICS CODE END ---

def boundary_loss(model, device):
    # --- INJECTED BOUNDARY CODE START ---
    {boundary_code}
    # --- INJECTED BOUNDARY CODE END ---

def train_and_evaluate(seed=42, round_num=0, plot_filename="pinn_result.png"):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINN().to(device)
    
    # 【关键修改】在训练循环开始前，生成一次固定的采样点
    # 这模仿了你简单代码里的行为
    x_fixed_raw = torch.rand(N_COL, IN_DIM, device=device)
    x_fixed_val = (DOMAIN_MAX - DOMAIN_MIN) * x_fixed_raw + DOMAIN_MIN
    x_fixed = x_fixed_val.detach() # 这是一个固定的 Tensor

    print(f"--- Round {round_num} Start Training (Opt: {OPTIMIZER}, Epochs: {EPOCHS}) ---")

    opt = None
    scheduler = None

    if OPTIMIZER == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        
        if LR_SCHEDULER_TYPE == "step":
            step_size = max(1, int(EPOCHS / 3))
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=LR_DECAY_GAMMA)
        elif LR_SCHEDULER_TYPE == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)
            
        for epoch in range(EPOCHS):
            opt.zero_grad()
            # 【关键修改】传入固定的 x_fixed
            loss = physics_loss(model, x_fixed) + BC_WEIGHT * boundary_loss(model, device)
            loss.backward()
            opt.step()
            
            if scheduler:
                scheduler.step()
            
            if epoch % 1000 == 0 or epoch == EPOCHS - 1:
                print(f"[Adam] Epoch {epoch:5d}/{EPOCHS} | Loss: {loss.item():.6e}")
            
    elif OPTIMIZER == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=EPOCHS, max_eval=EPOCHS*1.25, 
                                history_size=50, line_search_fn="strong_wolfe")
        lbfgs_iter = 0
        def closure():
            nonlocal lbfgs_iter
            opt.zero_grad()
            # 【关键修改】传入固定的 x_fixed
            loss = physics_loss(model, x_fixed) + BC_WEIGHT * boundary_loss(model, device)
            loss.backward()
            if lbfgs_iter % 1000 == 0:
                print(f"[LBFGS] Iter {lbfgs_iter:5d} | Loss: {loss.item():.6e}")
            lbfgs_iter += 1
            return loss
        opt.step(closure)

    # Evaluation & Plotting (保持不变)
    mse = -1.0 
    mae = -1.0
    with torch.no_grad():
        xs_plot = torch.linspace(DOMAIN_MIN, DOMAIN_MAX, 200, device=device).unsqueeze(1)
        # 如果是 2D (PDE), 这里 plotting_code 需要特殊处理，这里假设是 ODE 或 plotting_code 自带逻辑
        # 为了兼容 Problems.py 里的绘图代码，我们需要确保变量名一致
        pred_plot = model(xs_plot)
        
        plt.figure(figsize=(8, 5))
        # --- INJECTED PLOTTING CODE START ---
        {plotting_code}
        # --- INJECTED PLOTTING CODE END ---
        plt.legend()
        plt.title(f"Round {round_num} Result (Epochs: {EPOCHS})\\nMSE: {mse:.2e}")
        plt.grid(True)
        image_path = f"pinn_round_{round_num}_seed_{seed}.png"
        plt.savefig(image_path)
        plt.close()

    final_loss = physics_loss(model, x_fixed) + BC_WEIGHT * boundary_loss(model, device)
    print(f"--- Round {round_num} Finished. Final Loss: {final_loss.item():.6e} ---\n")
        
    return {
        "final_loss": final_loss.item(), 
        "mse": mse, 
        "mae": mae, 
        "image_path": image_path
    }
"""