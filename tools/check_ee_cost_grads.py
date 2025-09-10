#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from mpcpanda import PinocchioPandaDynamics, EETrackingCostFn


def random_rotation_matrix():
    # Random axis-angle
    axis = np.random.randn(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    angle = np.random.uniform(-np.pi, np.pi)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def scalar_loss(C, c):
    # Simple smooth scalar objective from outputs
    return 0.5 * (C.pow(2).sum() + c.pow(2).sum())


def finite_diff_grad(func, param, eps=1e-6):
    """Compute FD gradient of scalar func() wrt tensor 'param' in-place.

    func returns a scalar torch tensor. 'param' is a leaf tensor.
    """
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

    urdf_path = os.path.join(REPO_ROOT, 'robot_description', 'panda_with_gripper.urdf')
    dt = 0.01
    dyn = PinocchioPandaDynamics(urdf_path=urdf_path, dt=dt, device=device, with_gravity=True).to(device)

    n_state = 14
    n_ctrl = 7
    ee_frame_id = dyn.model.getFrameId('panda_hand')

    # Random test point
    torch.manual_seed(0)
    np.random.seed(0)
    q = torch.zeros(7, device=device, dtype=torch.float64)
    ee_goal_pos = torch.tensor([0.4, 0.2, 0.5], device=device, dtype=torch.float64, requires_grad=True)
    ee_goal_R_np = random_rotation_matrix()
    ee_goal_R = torch.tensor(ee_goal_R_np, device=device, dtype=torch.float64, requires_grad=True)

    pos_w = torch.tensor(5.0, device=device, dtype=torch.float64, requires_grad=True)
    ori_w = torch.tensor(1.0, device=device, dtype=torch.float64, requires_grad=True)
    v_w = torch.tensor(1e-2, device=device, dtype=torch.float64, requires_grad=True)
    u_w = torch.tensor(1e-9, device=device, dtype=torch.float64, requires_grad=True)

    # Wrap creator for autograd graph
    def forward_and_loss():
        C, c = EETrackingCostFn.apply(q, ee_goal_pos, ee_goal_R, pos_w, ori_w, v_w, u_w,
                                      dyn, ee_frame_id, n_state, n_ctrl)
        return scalar_loss(C, c)

    # Autograd gradients
    loss = forward_and_loss()
    grads = torch.autograd.grad(loss, [ee_goal_pos, ee_goal_R, pos_w, ori_w, v_w, u_w], retain_graph=True, allow_unused=True)
    g_auto = {
        'ee_goal_pos': grads[0],
        'ee_goal_R': grads[1],
        'pos_w': grads[2],
        'ori_w': grads[3],
        'v_w': grads[4],
        'u_w': grads[5],
    }

    print(g_auto)


    # Finite-difference gradients
    def loss_func_factory():
        return forward_and_loss()

    g_fd = {}
    g_fd['ee_goal_pos'] = finite_diff_grad(loss_func_factory, ee_goal_pos)
    g_fd['pos_w'] = finite_diff_grad(loss_func_factory, pos_w)
    g_fd['ori_w'] = finite_diff_grad(loss_func_factory, ori_w)
    g_fd['v_w'] = finite_diff_grad(loss_func_factory, v_w)
    g_fd['u_w'] = finite_diff_grad(loss_func_factory, u_w)

    def report(name):
        a = g_auto[name]
        f = g_fd[name]
        if a is None:
            print(f"{name}: autograd=None, fd max {f.abs().max().item():.3e}, mean {f.abs().mean().item():.3e}")
        else:
            err = (a - f).abs()
            print(f"{name}: max|auto-fd| {err.max().item():.3e}, mean {err.mean().item():.3e}")

    print("EETrackingCostFn gradient check vs finite differences:")
    for key in ['ee_goal_pos', 'ee_goal_R', 'pos_w', 'ori_w', 'v_w', 'u_w']:
        if key == 'ee_goal_R':
            # Manifold-consistent directional check for rotation target
            # Basis twists along x,y,z at goal rotation
            def hat(v):
                return torch.tensor([[0, -v[2], v[1]],
                                     [v[2], 0, -v[0]],
                                     [-v[1], v[0], 0]], dtype=ee_goal_R.dtype)
            basis = [torch.tensor([1.0,0.0,0.0], dtype=ee_goal_R.dtype),
                     torch.tensor([0.0,1.0,0.0], dtype=ee_goal_R.dtype),
                     torch.tensor([0.0,0.0,1.0], dtype=ee_goal_R.dtype)]

            eps = 1e-6
            auto = g_auto['ee_goal_R']
            errs = []
            for i, b in enumerate(basis):
                Bi = ee_goal_R @ hat(b)
                # directional derivative from autograd: <G, Bi>
                dir_auto = (auto * Bi).sum().item()
                # FD directional derivative via group perturbation
                with torch.no_grad():
                    R_orig = ee_goal_R.clone()
                    # plus
                    ee_goal_R.copy_(R_orig @ torch.matrix_exp(eps * hat(b)))
                loss_plus = forward_and_loss().item()
                with torch.no_grad():
                    ee_goal_R.copy_(R_orig @ torch.matrix_exp(-eps * hat(b)))
                loss_minus = forward_and_loss().item()
                with torch.no_grad():
                    ee_goal_R.copy_(R_orig)
                dir_fd = (loss_plus - loss_minus) / (2.0 * eps)
                errs.append(abs(dir_auto - dir_fd))
            print(f"ee_goal_R (tangent dirs): max|auto-fd| {max(errs):.3e}, mean {np.mean(errs):.3e}")
        else:
            report(key)

    # Note: By design, current EETrackingCostFn has zero gradient w.r.t ee_goal_R
    # (it detaches in forward). The FD will likely be non-zero; this highlights
    # the missing analytical gradient path for orientation target.

if __name__ == '__main__':
    main()
