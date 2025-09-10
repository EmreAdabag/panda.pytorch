#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'mpc.pytorch'))

from mpcpanda import PinocchioPandaDynamics, make_build_ee_tracking_cost
from mpc.mpc import MPC, QuadCost, GradMethods
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime


def random_rotation_matrix():
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.random.uniform(-np.pi, np.pi)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def scalar_loss_from_mpc(x_seq, u_seq):
    # Simple smooth scalar based on MPC trajectories
    # Use a mix to ensure dependencies on both states and controls
    return 0.5 * (x_seq.pow(2).mean() + 1e-2 * u_seq.pow(2).mean())


def finite_diff_grad(func, param, eps=1e-6):
    g = torch.zeros_like(param)
    it = np.nditer(np.arange(param.numel()))
    while not it.finished:
        idx = int(it[0])
        with torch.no_grad():
            flat = param.view(-1)
            orig = flat[idx].item()
            flat[idx] = orig + eps
        loss_plus = func().item()
        with torch.no_grad():
            flat = param.view(-1)
            flat[idx] = orig - eps
        loss_minus = func().item()
        with torch.no_grad():
            flat = param.view(-1)
            flat[idx] = orig
        g.view(-1)[idx] = torch.tensor((loss_plus - loss_minus) / (2.0 * eps), dtype=param.dtype, device=param.device)
        it.iternext()
    return g


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')

    # Dynamics
    urdf_path = os.path.join(REPO_ROOT, 'robot_description', 'panda_with_gripper.urdf')
    dt = 0.01
    dyn = PinocchioPandaDynamics(urdf_path=urdf_path, dt=dt, device=device, with_gravity=True).to(device)

    n = dyn.nv
    n_state = dyn.nq + dyn.nv
    n_ctrl = dyn.nv

    # MPC setup
    T = 10
    mpc = MPC(
        n_state=n_state,
        n_ctrl=n_ctrl,
        T=T,
        u_lower=None,
        u_upper=None,
        lqr_iter=2,
        verbose=0,
        grad_method=GradMethods.ANALYTIC,
        exit_unconverged=False,
        detach_unconverged=False,
        eps=1e-3,
        n_batch=1,
    )

    # EE tracking cost builder
    ee_frame_id = dyn.model.getFrameId('panda_hand')
    cost_builder = make_build_ee_tracking_cost(dyn, ee_frame_id, n_state, n_ctrl)

    # Initial state
    torch.manual_seed(0)
    np.random.seed(0)
    q0 = torch.zeros(n, device=device, dtype=torch.float64)
    v0 = torch.zeros(n, device=device, dtype=torch.float64)
    x_init = torch.cat([q0, v0], dim=0)[None, :]  # [1, n_state]

    # Parameters to differentiate
    ee_goal_pos = torch.tensor([0.4, 0.2, 0.5], device=device, dtype=torch.float64, requires_grad=True)
    ee_goal_R = torch.tensor(random_rotation_matrix(), device=device, dtype=torch.float64, requires_grad=True)
    pos_w = torch.tensor(5.0, device=device, dtype=torch.float64, requires_grad=True)
    ori_w = torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True)
    v_w = torch.tensor(1e-2, device=device, dtype=torch.float64, requires_grad=True)
    u_w = torch.tensor(1e-9, device=device, dtype=torch.float64, requires_grad=True)

    # Forward solve wrapper to keep graph
    def forward_and_loss():
        cost = cost_builder(
            q=q0,
            ee_goal_pos=ee_goal_pos,
            ee_goal_R=ee_goal_R,
            pos_weight=pos_w,
            orient_weight=ori_w,
            v_weight=v_w,
            u_weight=u_w,
        )
        x_mpc, u_mpc, obj = mpc(x_init, cost, dyn)
        return scalar_loss_from_mpc(x_mpc, u_mpc)

    # Prepare output directory for plots
    out_dir = os.path.join(REPO_ROOT, 'tools', 'mpc_grad_plots')
    os.makedirs(out_dir, exist_ok=True)

    # Helper: sweep and plot for scalar tensor param
    def sweep_scalar_param(name, param, base_val, delta=1e-1, N=21, h=1e-6):
        deltas = np.linspace(-delta, delta, N)
        fd_vals = []
        ag_vals = []
        for d in deltas:
            with torch.no_grad():
                param.copy_(torch.tensor(base_val + d, dtype=param.dtype, device=param.device))
            # autograd gradient at this point
            loss = forward_and_loss()
            g = torch.autograd.grad(loss, param, retain_graph=False, create_graph=False, allow_unused=True)
            g_val = g[0].item() if g[0] is not None else np.nan
            ag_vals.append(g_val)
            # finite-difference directional derivative d/dparam via central diff
            with torch.no_grad():
                param.add_(h)
            lp = forward_and_loss().item()
            with torch.no_grad():
                param.add_(-2*h)
            lm = forward_and_loss().item()
            with torch.no_grad():
                param.add_(h)  # restore
            fd = (lp - lm) / (2*h)
            fd_vals.append(fd)
        # restore
        with torch.no_grad():
            param.copy_(torch.tensor(base_val, dtype=param.dtype, device=param.device))

        # plot
        plt.figure(figsize=(6,4))
        plt.plot(deltas, fd_vals, 'o', label='FD')
        plt.plot(deltas, ag_vals, '-', label='Autograd')
        plt.xlabel(f'{name} offset')
        plt.ylabel(f'dL/d{name}')
        plt.title(f'Gradient sweep: {name}')
        plt.legend()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(out_dir, f'{ts}_{name}_sweep.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f'Saved {fname}')

    # Helper: sweep and plot for vector component
    def sweep_vector_component(name, vec, idx, base_val, delta=5e-2, N=21, h=1e-6):
        deltas = np.linspace(-delta, delta, N)
        fd_vals = []
        ag_vals = []
        for d in deltas:
            with torch.no_grad():
                vec[idx] = base_val + d
            loss = forward_and_loss()
            g = torch.autograd.grad(loss, vec, retain_graph=False, create_graph=False, allow_unused=True)
            g_val = (g[0][idx].item() if g[0] is not None else np.nan)
            ag_vals.append(g_val)
            with torch.no_grad():
                vec[idx] += h
            lp = forward_and_loss().item()
            with torch.no_grad():
                vec[idx] -= 2*h
            lm = forward_and_loss().item()
            with torch.no_grad():
                vec[idx] += h
            fd_vals.append((lp - lm) / (2*h))
        with torch.no_grad():
            vec[idx] = base_val

        plt.figure(figsize=(6,4))
        plt.plot(deltas, fd_vals, 'o', label='FD')
        plt.plot(deltas, ag_vals, '-', label='Autograd')
        plt.xlabel(f'{name}[{idx}] offset')
        plt.ylabel(f'dL/d{name}[{idx}]')
        plt.title(f'Gradient sweep: {name}[{idx}]')
        plt.legend()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(out_dir, f'{ts}_{name}_{idx}_sweep.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f'Saved {fname}')

    # Helper: sweep along SO(3) axis direction and plot directional derivative
    def hat(v):
        H = torch.zeros(3, 3, dtype=v.dtype, device=v.device)
        H[0,1] = -v[2]; H[0,2] =  v[1]
        H[1,0] =  v[2]; H[1,2] = -v[0]
        H[2,0] = -v[1]; H[2,1] =  v[0]
        return H

    def sweep_rotation_axis(axis_name, axis_vec, angle_delta=5e-2, N=21, h=1e-6):
        deltas = np.linspace(-angle_delta, angle_delta, N)
        fd_vals = []
        ag_vals = []
        R_base = ee_goal_R.detach().clone()
        for d in deltas:
            with torch.no_grad():
                ee_goal_R.copy_(R_base @ torch.matrix_exp(d * hat(axis_vec)))
            # autograd directional derivative dL/dd = <dL/dR, dR/dd>
            loss = forward_and_loss()
            G = torch.autograd.grad(loss, ee_goal_R, retain_graph=False, create_graph=False, allow_unused=True)[0]
            if G is None:
                ag_vals.append(np.nan)
            else:
                Bi = ee_goal_R @ hat(axis_vec)
                ag_vals.append((G * Bi).sum().item())
            # finite diff in angle
            with torch.no_grad():
                ee_goal_R.copy_(R_base @ torch.matrix_exp((d + h) * hat(axis_vec)))
            lp = forward_and_loss().item()
            with torch.no_grad():
                ee_goal_R.copy_(R_base @ torch.matrix_exp((d - h) * hat(axis_vec)))
            lm = forward_and_loss().item()
            fd_vals.append((lp - lm) / (2*h))
        with torch.no_grad():
            ee_goal_R.copy_(R_base)

        plt.figure(figsize=(6,4))
        plt.plot(deltas, fd_vals, 'o', label='FD')
        plt.plot(deltas, ag_vals, '-', label='Autograd')
        plt.xlabel(f'angle offset about {axis_name} [rad]')
        plt.ylabel('dL/d(angle)')
        plt.title(f'Rotation gradient sweep: {axis_name}-axis')
        plt.legend()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = os.path.join(out_dir, f'{ts}_R_{axis_name}_sweep.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f'Saved {fname}')

    # Run sweeps and generate plots
    # Vector ee_goal_pos components
    for i, nm in enumerate(['x','y','z']):
        sweep_vector_component('ee_goal_pos', ee_goal_pos, i, ee_goal_pos[i].item())

    # Scalar weights
    sweep_scalar_param('pos_w', pos_w, pos_w.item())
    sweep_scalar_param('ori_w', ori_w, ori_w.item())
    sweep_scalar_param('v_w', v_w, v_w.item(), delta=5e-3)
    sweep_scalar_param('u_w', u_w, u_w.item(), delta=5e-10, h=1e-12)

    # Rotation axes sweeps
    basis = [
        ('x', torch.tensor([1.0, 0.0, 0.0], dtype=ee_goal_R.dtype, device=ee_goal_R.device)),
        ('y', torch.tensor([0.0, 1.0, 0.0], dtype=ee_goal_R.dtype, device=ee_goal_R.device)),
        ('z', torch.tensor([0.0, 0.0, 1.0], dtype=ee_goal_R.dtype, device=ee_goal_R.device)),
    ]
    for axis_name, axis_vec in basis:
        sweep_rotation_axis(axis_name, axis_vec)


if __name__ == '__main__':
    main()
