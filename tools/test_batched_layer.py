import torch
import os
from scipy.spatial.transform import Rotation as R
import time 

# Add paths
import sys
sys.path.insert(0, './')
from mpcpanda import PandaEETrackingMPCLayer


def make_quat(rpy):
    rot = R.from_euler('xyz', rpy.tolist())
    # SciPy returns as [x, y, z, w]
    return torch.tensor(rot.as_quat(), dtype=torch.get_default_dtype())


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    print(f'using device: {device}')

    # Config
    B = int(os.environ.get('BATCH', '128'))
    K = int(os.environ.get('K', '2'))
    T = int(os.environ.get('T', '16'))
    dt = float(os.environ.get('DT', '0.01'))
    urdf_path = os.environ.get('URDF', 'robot_description/panda_with_gripper.urdf')

    # Layer with absolute goal schedule [K]
    # Spread absolute goal changes every 40 steps by default
    goal_ts_abs = torch.tensor([5,T])
    layer = PandaEETrackingMPCLayer(
        urdf_path=urdf_path,
        T=T,
        goal_timesteps_abs=goal_ts_abs,
        dt=dt,
        device=device,
        with_gravity=True,
        lqr_iter=1,
        eps=1e-3,
        verbose=0,
    ).to(device)

    n_state = layer.n_state
    n_ctrl = layer.n_ctrl

    # Random-ish initial states
    q0 = torch.zeros(B, n_ctrl, device=device)
    v0 = torch.zeros(B, n_ctrl, device=device)
    x_init = torch.cat([q0, v0], dim=-1)

    # Goals per batch and per K
    goal_positions = torch.zeros(B, K, 3, device=device)
    goal_quaternions = torch.zeros(B, K, 4, device=device)
    for b in range(B):
        for k in range(K):
            # Space out positions a bit per batch and per k
            goal_positions[b, k] = torch.tensor([0.4 + 0.05*b, 0.1*(-1)**k, 0.45 + 0.02*k], device=device)
            rpy = torch.tensor([3.14 - 0.5*k, 0.0, 0.0], device=device)
            goal_quaternions[b, k] = make_quat(rpy).to(device)

    # Enable gradients on goals (they are leaves)
    goal_positions.requires_grad_()
    goal_quaternions.requires_grad_()

    # Scalar weights (enable gradient to test backprop)
    pos_w = torch.tensor(4.0, dtype=torch.get_default_dtype(), device=device, requires_grad=True)
    ori_w = torch.tensor(2.0, dtype=torch.get_default_dtype(), device=device, requires_grad=True)
    v_w = torch.tensor(1e-2, dtype=torch.get_default_dtype(), device=device, requires_grad=True)
    u_w = torch.tensor(1e-2, dtype=torch.get_default_dtype(), device=device, requires_grad=True)

    # Warmup pass
    x_traj, u_traj, costs = layer(x_init, goal_positions, goal_quaternions, pos_w, ori_w, v_w, u_w)
    obj = (u_traj.pow(2).sum()) + (x_traj.pow(2).sum())
    obj.backward()
    print('Shapes:', x_traj.shape, u_traj.shape, costs.shape)
    # print('Gradients:')
    # print('x_init.grad', x_init.grad)
    # print('goal_position.grad', goal_positions.grad)
    # print('goal_quaternions.grad', goal_quaternions.grad)
    # print('pos_w.grad:', pos_w.grad)
    # print('ori_w.grad:', ori_w.grad)
    # print('v_w.grad:', v_w.grad)
    # print('u_w.grad:', u_w.grad)

    # time forward
    s = time.monotonic()
    torch.cuda.synchronize()
    for _ in range(10):
        x_traj, _, _ = layer(x_init, goal_positions, goal_quaternions, pos_w, ori_w, v_w, u_w)
    torch.cuda.synchronize()
    e = time.monotonic()
    print(f"forward time: {(e - s) * 100} ms")

    # time backward
    s = time.monotonic()
    torch.cuda.synchronize()
    for _ in range(10):
        x_traj, u_traj, _ = layer(x_init, goal_positions, goal_quaternions, pos_w, ori_w, v_w, u_w)
        obj = (u_traj.pow(2).sum()) + (x_traj.pow(2).sum())
        obj.backward()
    torch.cuda.synchronize()
    e = time.monotonic()
    print(f"backward time: {(e - s) * 100} ms")



if __name__ == '__main__':
    main()
