#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'mpc.pytorch'))

from mpcpanda import PandaEETrackingMPCLayer
from mpc.mpc import GradMethods


def random_quaternion_xyzw():
    v = np.random.randn(4)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def scalar_loss_from_mpc(x_seq, u_seq):
    return 0.5 * (x_seq.pow(2).mean() + 1e-2 * u_seq.pow(2).mean())


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device('cpu')

    # Dynamics
    urdf_path = os.path.join(REPO_ROOT, 'robot_description', 'panda_with_gripper.urdf')
    dt = 0.01
    # MPC layer
    T = 10
    goal_ts_abs = torch.tensor([3, 9], dtype=torch.long, device=device)
    layer = PandaEETrackingMPCLayer(
        urdf_path=urdf_path,
        T=T,
        goal_timesteps_abs=goal_ts_abs,
        dt=dt,
        device=device,
        with_gravity=True,
        lqr_iter=2,
        verbose=0,
        eps=1e-3,
    ).to(device)

    n = layer.n_ctrl
    n_state = layer.n_state

    # Multi-goal parameters (runtime, differentiable wrt poses)
    goal_positions = torch.tensor(
        [
            [0.40, 0.20, 0.50],
            [0.55,-0.20, 0.45],
        ], dtype=torch.float64, device=device, requires_grad=True
    )  # [K,3]
    quats_np = np.stack([
        random_quaternion_xyzw(),
        random_quaternion_xyzw(),
    ], axis=0)
    goal_quaternions = torch.tensor(quats_np, dtype=torch.float64, device=device, requires_grad=True)  # [K,4]

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
        # Ensure consistent goal schedule across evaluations (avoid internal step drift)
        layer.reset_schedule(0)
        x_mpc, u_mpc, _ = layer(
            x_init,
            goal_positions.unsqueeze(0),
            goal_quaternions.unsqueeze(0),
            pos_w,
            ori_w,
            v_w,
            u_w,
        )
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

    print("\nChecking gradients wrt goal_quaternions (axis-angle FD vs autograd directional):")
    h_ang = 1e-6
    basis = [
        ('x', torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device)),
        ('y', torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=device)),
        ('z', torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=device)),
    ]
    loss = forward_and_loss()
    g_quat = torch.autograd.grad(loss, goal_quaternions, retain_graph=False, create_graph=False)[0]

    def quat_mul(q, r):
        x1,y1,z1,w1 = q
        x2,y2,z2,w2 = r
        return torch.tensor([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=q.dtype, device=q.device)

    for k in range(goal_quaternions.size(0)):
        q_base = goal_quaternions[k].detach().clone()
        for axis_name, axis_vec in basis:
            # Autograd directional derivative dL/dtheta = <dL/dq, dq/dtheta>
            # dq/dtheta at 0 for right-mult by exp(0.5*theta*axis): 0.5 * q ⊗ [axis, 0]
            J_dir = 0.5 * quat_mul(q_base, torch.tensor([axis_vec[0], axis_vec[1], axis_vec[2], 0.0], dtype=q_base.dtype, device=q_base.device))
            ag = (g_quat[k] * J_dir).sum().item()

            # Finite difference via small rotation quaternion on the right
            dq = torch.tensor([
                axis_vec[0]*np.sin(h_ang/2.0),
                axis_vec[1]*np.sin(h_ang/2.0),
                axis_vec[2]*np.sin(h_ang/2.0),
                np.cos(h_ang/2.0),
            ], dtype=q_base.dtype, device=q_base.device)
            with torch.no_grad():
                goal_quaternions[k].copy_(quat_mul(q_base, dq))
            lp = forward_and_loss().item()
            with torch.no_grad():
                goal_quaternions[k].copy_(quat_mul(q_base, torch.tensor([-dq[0], -dq[1], -dq[2], dq[3]], dtype=q_base.dtype, device=q_base.device)))
            lm = forward_and_loss().item()
            with torch.no_grad():
                goal_quaternions[k].copy_(q_base)
            fd = (lp - lm) / (2*h_ang)
            rel = abs(ag - fd) / (abs(fd) + 1e-12)
            print(f"  goal_quat[{k}] axis {axis_name}: autograd={ag:.6e}  FD={fd:.6e}  rel_err={rel:.2e}")

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
