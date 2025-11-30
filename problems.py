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
        
        "physics_code": """
    mu = 1.0
    dx_dt = autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    d2x_dt2 = autograd.grad(dx_dt, x, torch.ones_like(dx_dt), create_graph=True)[0]
    residue = d2x_dt2 - mu * (1 - y**2) * dx_dt + y
    return (residue**2).mean()
""",
        "boundary_code": """
    t0 = torch.tensor([[0.0]], device=device).requires_grad_(True)
    x0 = model(t0)
    dx0_dt = autograd.grad(x0, t0, torch.ones_like(x0), create_graph=True)[0]
    return ((x0 - 2.0)**2 + (dx0_dt - 0.0)**2).mean()
""",
        # 【修改点】在这里加入 Scipy 真值生成 + MSE/MAE 计算
        "plotting_code": """
        from scipy.integrate import solve_ivp
        
        # 1. 使用 Scipy 求解真值
        def vdp_sys(t, z):
            mu = 1.0
            x, dx = z
            return [dx, mu*(1-x**2)*dx - x]

        t_eval = xs_plot.cpu().numpy().flatten()
        # 求解 IVP
        sol = solve_ivp(vdp_sys, [0, 10], [2, 0], t_eval=t_eval)
        true_vals_np = sol.y[0] # x(t)
        
        # 转回 Tensor 以便计算 MSE
        true_plot = torch.tensor(true_vals_np, dtype=torch.float32, device=device).unsqueeze(1)
        
        # 2. 计算指标
        mse = torch.mean((pred_plot - true_plot)**2).item()
        mae = torch.mean(torch.abs(pred_plot - true_plot)).item()

        # 3. 绘图
        plt.plot(t_eval, true_vals_np, 'b-', label='Truth (RK45)', linewidth=2, alpha=0.6)
        plt.plot(xs_plot.cpu(), pred_plot.cpu(), 'r--', label='PINN', linewidth=2)
        plt.xlabel("t"); plt.ylabel("x(t)")
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
        "input_dim": 2,  # 【关键】输入是 (t, x) 两个维度
        "output_dim": 1, # 输出是 u 一个维度
        "domain_range": [-1.0, 1.0], # 这里主要指空间 x 的范围，时间 t 通常在内部处理或归一化
        
        # 物理方程: u_t + u*u_x - nu*u_xx = 0
        "physics_code": """
    # x_in 的形状是 [N, 2]。
    # 我们约定: x_in[:, 0] 是 t (时间), x_in[:, 1] 是 x (空间)
    # (注意：这取决于你怎么生成数据，通常习惯 t在前或 x在前，这里假设 input=[t, x])
    
    u = y
    
    # 求一阶导数 (对 t 和 x)
    du_dinput = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    du_dt = du_dinput[:, 0:1]
    du_dx = du_dinput[:, 1:2]
    
    # 求二阶导数 (对 x)
    du_dxx = autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True)[0][:, 1:2]
    
    # Burgers 参数
    viscosity = 0.01 / math.pi
    
    # 残差
    res = du_dt + u * du_dx - viscosity * du_dxx
    return (res**2).mean()
""",
        
        # 边界条件 (BC) + 初始条件 (IC)
        "boundary_code": """
    # 这是一个混合 Loss，包含 IC (t=0) 和 BC (x=-1, x=1)
    
    # 1. IC: t=0, x in [-1, 1]. u(0,x) = -sin(pi*x)
    # 随机采样空间点
    x_space = torch.rand(N_COL // 2, 1, device=device) * 2.0 - 1.0 # [-1, 1]
    t_zero  = torch.zeros_like(x_space)
    # 拼接成 (t, x) 输入
    in_ic = torch.cat([t_zero, x_space], dim=1).requires_grad_(True)
    u_ic_pred = model(in_ic)
    u_ic_true = -torch.sin(math.pi * x_space)
    loss_ic = ((u_ic_pred - u_ic_true)**2).mean()
    
    # 2. BC: x=-1 和 x=1, t in [0, 1]. u(t, -1) = u(t, 1) = 0
    t_time = torch.rand(N_COL // 2, 1, device=device) # [0, 1]
    x_neg  = torch.ones_like(t_time) * -1.0
    x_pos  = torch.ones_like(t_time) * 1.0
    
    in_bc_neg = torch.cat([t_time, x_neg], dim=1)
    in_bc_pos = torch.cat([t_time, x_pos], dim=1)
    
    u_bc_neg = model(in_bc_neg)
    u_bc_pos = model(in_bc_pos)
    
    loss_bc = (u_bc_neg**2).mean() + (u_bc_pos**2).mean()
    
    return loss_ic + loss_bc
""",
        
        # 绘图: 画热力图比较炫酷，或者画几个时间切片
        "plotting_code": """
        # 为了简化，我们画 t=0.5 时刻的 u(x) 切片对比
        # 真值通常需要解析解或数值模拟，这里用近似解析解或跳过真值对比，
        # 或者我们只画 t=0.5 的预测曲线看是否平滑
        
        # 构造 t=0.5 的输入
        t_slice = 0.5
        x_eval = torch.linspace(-1, 1, 100, device=device).unsqueeze(1)
        t_eval = torch.ones_like(x_eval) * t_slice
        in_eval = torch.cat([t_eval, x_eval], dim=1)
        
        u_pred = model(in_eval)
        
        plt.plot(x_eval.cpu().numpy(), u_pred.cpu().numpy(), 'r-', label=f'PINN t={t_slice}')
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"Burgers Equation (Slice at t={t_slice})")
        
        # 对于 PDE，单点 MSE 很难算，通常只算 Loss
        # 这里给个占位符
        mse = 0.0
        mae = 0.0
"""
    }
}