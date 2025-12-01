# problems.py

PROBLEMS = {
    "poisson_1d": {
        "description": "Solve 1D Poisson: y''(x) = -pi^2 * sin(pi x) on [0,1], y(0)=0, y(1)=0.",
        "input_dim": 1,
        "output_dim": 1,
        "domain_range": [0.0, 1.0],
        
        "physics_code": """
    dy_dx = autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    d2y_dx2 = autograd.grad(dy_dx, x, torch.ones_like(dy_dx), create_graph=True)[0]
    target = -(math.pi**2) * torch.sin(math.pi * x)
    return ((d2y_dx2 - target)**2).mean()
""",
        "boundary_code": """
    x0 = torch.tensor([[0.0]], device=device)
    x1 = torch.tensor([[1.0]], device=device)
    y0, y1 = model(x0), model(x1)
    return (y0**2 + y1**2).mean()
""",
        # 【修改点】在这里加入 MSE/MAE 计算
        "plotting_code": """
        # 1. 计算真值
        true_plot = torch.sin(math.pi * xs_plot)
        
        # 2. 计算指标 (Tensor -> Float)
        mse = torch.mean((pred_plot - true_plot)**2).item()
        mae = torch.mean(torch.abs(pred_plot - true_plot)).item()

        # 3. 绘图
        plt.plot(xs_plot.cpu(), true_plot.cpu(), 'b-', label='Truth', linewidth=2, alpha=0.6)
        plt.plot(xs_plot.cpu(), pred_plot.cpu(), 'r--', label='PINN', linewidth=2)
        plt.xlabel("x"); plt.ylabel("y")
"""
    },

"van_der_pol": {
        "description": "Solve Van der Pol Oscillator: x'' - mu(1-x^2)x' + x = 0, mu=1.0. Range [0,10]. IC: x(0)=2, x'(0)=0.",
        "input_dim": 1,
        "output_dim": 1,
        "domain_range": [0.0, 10.0],
        
        # 物理残差 (保持不变，兼容 x_fixed)
        "physics_code": """
    mu = 1.0
    # x 和 y 由 template 传入
    dx_dt = autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    d2x_dt2 = autograd.grad(dx_dt, x, torch.ones_like(dx_dt), create_graph=True)[0]
    residue = d2x_dt2 - mu * (1 - y**2) * dx_dt + y
    return (residue**2).mean()
""",
        
        # 边界条件 (保持不变)
        "boundary_code": """
    # 强制 t=0 处的 x 和 x'
    t0 = torch.tensor([[0.0]], device=device).requires_grad_(True)
    x0 = model(t0)
    dx0_dt = autograd.grad(x0, t0, torch.ones_like(x0), create_graph=True)[0]
    return ((x0 - 2.0)**2 + (dx0_dt - 0.0)**2).mean()
""",
        
        # 绘图代码 (针对最新 template 优化)
        # 注意：因为 IN_DIM=1，template 会自动生成 xs_plot 和 pred_plot，我们可以直接用
        "plotting_code": """
        import numpy as np
        from scipy.integrate import solve_ivp
        
        # 1. 使用 Scipy 求解真值
        def vdp_sys(t, z):
            mu = 1.0
            x, dx = z
            return [dx, mu*(1-x**2)*dx - x]

        # xs_plot 已经是 template 生成好的 Tensor (200, 1)
        t_eval = xs_plot.cpu().numpy().flatten()
        
        # 求解 IVP
        sol = solve_ivp(vdp_sys, [0, 10], [2, 0], t_eval=t_eval)
        true_vals_np = sol.y[0] # x(t)
        
        # 转回 Tensor 以便计算 MSE
        true_plot = torch.tensor(true_vals_np, dtype=torch.float32, device=device).unsqueeze(1)
        
        # 2. 计算指标 (pred_plot 也是 template 生成好的)
        mse = torch.mean((pred_plot - true_plot)**2).item()
        mae = torch.mean(torch.abs(pred_plot - true_plot)).item()

        # 3. 绘图
        plt.plot(t_eval, true_vals_np, 'b-', label='Truth (RK45)', linewidth=2, alpha=0.6)
        plt.plot(xs_plot.cpu(), pred_plot.cpu(), 'r--', label='PINN', linewidth=2)
        plt.xlabel("t")
        plt.ylabel("x(t)")
        plt.title(f"Van der Pol (MSE: {mse:.2e})")
"""
    },
"pendulum": {
        "description": "Non-linear Pendulum: y'' + sin(y) = 0. Domain [0, 10]. IC: y(0)=2.0, y'(0)=0.",
        "input_dim": 1,
        "output_dim": 1,
        "domain_range": [0.0, 10.0],
        
        # 物理方程: y'' + sin(y) = 0 (假设 g/L = 1)
        "physics_code": """
    dy_dt = autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    d2y_dt2 = autograd.grad(dy_dt, x, torch.ones_like(dy_dt), create_graph=True)[0]
    
    # Residual = y'' + sin(y)
    residue = d2y_dt2 + torch.sin(y)
    return (residue**2).mean()
""",
        
        # 边界条件: y(0) = 2.0, y'(0) = 0
        "boundary_code": """
    t0 = torch.tensor([[0.0]], device=device).requires_grad_(True)
    y0 = model(t0)
    dy0_dt = autograd.grad(y0, t0, torch.ones_like(y0), create_graph=True)[0]
    
    # Loss = (y(0) - 2)^2 + (y'(0) - 0)^2
    return ((y0 - 2.0)**2 + (dy0_dt)**2).mean()
""",
        
        # 绘图与真值生成 (使用 Scipy 数值积分)
        "plotting_code": """
        from scipy.integrate import solve_ivp
        
        # 1. 定义 ODE 系统: [y, v] -> [v, -sin(y)]
        def pendulum_sys(t, z):
            return [z[1], -np.sin(z[0])]

        # 2. 生成真值
        t_eval = xs_plot.cpu().numpy().flatten()
        sol = solve_ivp(pendulum_sys, [0, 10], [2.0, 0.0], t_eval=t_eval)
        true_vals_np = sol.y[0]
        true_plot = torch.tensor(true_vals_np, dtype=torch.float32, device=device).unsqueeze(1)

        # 3. 计算指标 (MSE/MAE)
        mse = torch.mean((pred_plot - true_plot)**2).item()
        mae = torch.mean(torch.abs(pred_plot - true_plot)).item()

        # 4. 绘图
        plt.plot(t_eval, true_vals_np, 'b-', label='Truth (RK45)', linewidth=2, alpha=0.6)
        plt.plot(xs_plot.cpu(), pred_plot.cpu(), 'r--', label='PINN Prediction', linewidth=2)
        plt.xlabel("Time (t)")
        plt.ylabel("Angle (rad)")
"""
    },
"burgers_1d": {
        "description": "1D Burgers Eq: u_t + u*u_x - (0.01/pi)*u_xx = 0. Domain: x[-1,1], t[0,1].",
        "input_dim": 2,
        "output_dim": 1,
        "domain_range": [-1.0, 1.0],
        
        # 物理 Loss (保持不变)
        "physics_code": """
    u = y
    du_dinput = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    du_dt = du_dinput[:, 0:1]
    du_dx = du_dinput[:, 1:2]
    du_dxx = autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0][:, 1:2]
    viscosity = 0.01 / math.pi
    res = du_dt + u * du_dx - viscosity * du_dxx
    return (res**2).mean()
""",
        
        # 边界 Loss (保持不变)
        "boundary_code": """
    x_space = torch.rand(N_COL // 2, 1, device=device) * 2.0 - 1.0
    t_zero  = torch.zeros_like(x_space)
    in_ic = torch.cat([t_zero, x_space], dim=1).requires_grad_(True)
    u_ic_pred = model(in_ic)
    u_ic_true = -torch.sin(math.pi * x_space)
    loss_ic = ((u_ic_pred - u_ic_true)**2).mean()
    
    t_time = torch.rand(N_COL // 2, 1, device=device)
    x_neg  = torch.ones_like(t_time) * -1.0
    x_pos  = torch.ones_like(t_time) * 1.0
    in_bc_neg = torch.cat([t_time, x_neg], dim=1)
    in_bc_pos = torch.cat([t_time, x_pos], dim=1)
    u_bc_neg = model(in_bc_neg)
    u_bc_pos = model(in_bc_pos)
    loss_bc = (u_bc_neg**2).mean() + (u_bc_pos**2).mean()
    return loss_ic + loss_bc
""",

"plotting_code": """
        import numpy as np
        from scipy.integrate import solve_ivp

        # 1. 准备网格
        N_grid = 256  #稍微增加一点网格密度
        x_np = np.linspace(-1, 1, N_grid)
        dx = x_np[1] - x_np[0]
        viscosity = 0.01 / np.pi

        # 2. 定义 ODE 系统 (使用稳定的 Upwind 迎风格式)
        def burgers_rhs(t, u):
            # 边界条件 u[-1]=0, u[1]=0
            u_padded = np.pad(u, (1,1), mode='constant', constant_values=0)
            
            u_center = u_padded[1:-1]
            u_left   = u_padded[:-2]
            u_right  = u_padded[2:]
            
            # --- 关键修改：迎风格式 (Upwind Scheme) ---
            # 如果流速 u > 0，信息从左边来，用后向差分 (u_center - u_left)
            # 如果流速 u < 0，信息从右边来，用前向差分 (u_right - u_center)
            dudx_back = (u_center - u_left) / dx
            dudx_fwd  = (u_right - u_center) / dx
            
            # np.where 选择合适的差分方向
            convection = np.where(u_center > 0, u_center * dudx_back, u_center * dudx_fwd)
            
            # 扩散项依然用中心差分 (它是稳定的)
            diffusion = viscosity * (u_right - 2*u_center + u_left) / (dx**2)
            
            return -convection + diffusion

        # 3. 初始条件
        u0 = -np.sin(np.pi * x_np)
        
        # 4. 求解
        t_target = 0.5
        # 既然用了迎风格式，普通的 RK45 (默认) 其实就够稳了，BDF 也行
        sol = solve_ivp(burgers_rhs, [0, t_target], u0, method='RK45')
        u_true = sol.y[:, -1]
        
        # 5. PINN 预测
        x_eval = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(1)
        t_eval = torch.ones_like(x_eval) * t_target
        in_eval = torch.cat([t_eval, x_eval], dim=1)
        
        model.eval()
        u_pred = model(in_eval).detach().cpu().numpy().flatten()
        
        # 6. 计算指标
        mse = np.mean((u_true - u_pred)**2)
        mae = np.mean(np.abs(u_true - u_pred))

        # 7. 绘图
        plt.ylim(-1.2, 1.2)
        plt.plot(x_np, u_true, 'b-', label='Truth (Upwind)', linewidth=2, alpha=0.6)
        plt.plot(x_np, u_pred, 'r--', label=f'PINN (t={t_target})', linewidth=2)
        plt.title(f"Burgers Eq t={t_target} (MSE: {mse:.2e})")
        plt.xlabel("x"); plt.ylabel("u")
"""
    }
}