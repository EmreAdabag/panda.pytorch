#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'mpc.pytorch'))

from mpcpanda import PinocchioPandaDynamics, build_ee_tracking_cost_runtime
from mpc.mpc import MPC, QuadCost, GradMethods


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
    return 0.5 * (x_seq.pow(2).mean() + 1e-2 * u_seq.pow(2).mean())


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

    # Multi-goal parameters (runtime, differentiable wrt poses)
    goal_timesteps = torch.tensor([3, 9], dtype=torch.long, device=device)  # horizon-relative
    goal_positions = torch.tensor(
        [
            [0.40, 0.20, 0.50],
            [0.55,-0.20, 0.45],
        ], dtype=torch.float64, device=device, requires_grad=True
    )  # [K,3]
    goal_rotations = torch.tensor(
        [
            random_rotation_matrix(),
            random_rotation_matrix(),
        ], dtype=torch.float64, device=device, requires_grad=True
    )  # [K,3,3]

    # Initial state
    torch.manual_seed(0)
    np.random.seed(0)
    q0 = torch.zeros(n, device=device, dtype=torch.float64)
    v0 = torch.zeros(n, device=device, dtype=torch.float64)
    x_init = torch.cat([q0, v0], dim=0)[None, :]  # [1, n_state]

    # Parameters to differentiate (weights + all goal poses)
    pos_w = torch.tensor(5.0, device=device, dtype=torch.float64, requires_grad=True)
    ori_w = torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True)
    v_w = torch.tensor(1e-2, device=device, dtype=torch.float64, requires_grad=True)
    u_w = torch.tensor(1e-9, device=device, dtype=torch.float64, requires_grad=True)

    # Forward solve wrapper to keep graph
    def forward_and_loss():
        cost = build_ee_tracking_cost_runtime(
            q=q0,
            T=torch.tensor(T, dtype=torch.long, device=device),
            goal_timesteps=goal_timesteps,
            goal_positions=goal_positions,
            goal_rotations=goal_rotations,
            dynamics=dyn,
            pos_weight=pos_w,
            orient_weight=ori_w,
            v_weight=v_w,
            u_weight=u_w,
        )
        x_mpc, u_mpc, obj = mpc(x_init, cost, dyn)
        return scalar_loss_from_mpc(x_mpc, u_mpc)

    def hat(v):
        H = torch.zeros(3, 3, dtype=v.dtype, device=v.device)
        H[0,1] = -v[2]; H[0,2] =  v[1]
        H[1,0] =  v[2]; H[1,2] = -v[0]
        H[2,0] = -v[1]; H[2,1] =  v[0]
        return H

    # Console report: autograd vs finite differences
    print("Checking gradients wrt goal_positions (central FD vs autograd):")
    h_pos = 1e-6
    loss = forward_and_loss()
    g_pos = torch.autograd.grad(loss, goal_positions, retain_graph=True, create_graph=False)[0]
    for k in range(goal_positions.size(0)):
        for i in range(3):
            base = goal_positions[k, i].item()
            with torch.no_grad():
                goal_positions[k, i] = base + h_pos
            lp = forward_and_loss().item()
            with torch.no_grad():
                goal_positions[k, i] = base - h_pos
            lm = forward_and_loss().item()
            with torch.no_grad():
                goal_positions[k, i] = base
            fd = (lp - lm) / (2*h_pos)
            ag = g_pos[k, i].item()
            rel = abs(ag - fd) / (abs(fd) + 1e-12)
            print(f"  goal_pos[{k}][{i}]: autograd={ag:.6e}  FD={fd:.6e}  rel_err={rel:.2e}")

    print("\nChecking gradients wrt goal_rotations (axis-angle FD vs autograd directional):")
    h_ang = 1e-6
    basis = [
        ('x', torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device)),
        ('y', torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)),
        ('z', torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device)),
    ]
    loss = forward_and_loss()
    g_rot = torch.autograd.grad(loss, goal_rotations, retain_graph=False, create_graph=False)[0]
    for k in range(goal_rotations.size(0)):
        R_base = goal_rotations[k].detach().clone()
        for axis_name, axis_vec in basis:
            # Autograd directional derivative dL/dtheta = <dL/dR, R*hat(axis)>
            Bi = goal_rotations[k] @ hat(axis_vec)
            ag = (g_rot[k] * Bi).sum().item()
            with torch.no_grad():
                goal_rotations[k].copy_(R_base @ torch.matrix_exp(h_ang * hat(axis_vec)))
            lp = forward_and_loss().item()
            with torch.no_grad():
                goal_rotations[k].copy_(R_base @ torch.matrix_exp(-h_ang * hat(axis_vec)))
            lm = forward_and_loss().item()
            with torch.no_grad():
                goal_rotations[k].copy_(R_base)
            fd = (lp - lm) / (2*h_ang)
            rel = abs(ag - fd) / (abs(fd) + 1e-12)
            print(f"  goal_rot[{k}] axis {axis_name}: autograd={ag:.6e}  FD={fd:.6e}  rel_err={rel:.2e}")

    # Weights gradients
    print("\nChecking gradients wrt weights (central FD vs autograd):")
    def check_weight(name, param, h):
        loss = forward_and_loss()
        ag = torch.autograd.grad(loss, param, retain_graph=False, create_graph=False)[0].item()
        base = param.item()
        with torch.no_grad():
            param.copy_(torch.tensor(base + h, dtype=param.dtype, device=param.device))
        lp = forward_and_loss().item()
        with torch.no_grad():
            param.copy_(torch.tensor(base - h, dtype=param.dtype, device=param.device))
        lm = forward_and_loss().item()
        with torch.no_grad():
            param.copy_(torch.tensor(base, dtype=param.dtype, device=param.device))
        fd = (lp - lm) / (2*h)
        rel = abs(ag - fd) / (abs(fd) + 1e-12)
        print(f"  {name}: autograd={ag:.6e}  FD={fd:.6e}  rel_err={rel:.2e}")

    check_weight('pos_w', pos_w, 1e-6)
    check_weight('ori_w', ori_w, 1e-6)
    check_weight('v_w',  v_w,  1e-6)
    check_weight('u_w',  u_w,  1e-12)


if __name__ == '__main__':
    main()
