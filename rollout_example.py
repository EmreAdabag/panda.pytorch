import torch
import numpy as np
import pybullet as p
import pybullet_data
import sys
import os
import time
import numpy as np
import os
import pinocchio as pin
import pinocchio.rpy as rpy
from scipy.spatial.transform import Rotation as R


# Import MPC components and Panda dynamics from local mpc.py
from mpcpanda import (
    PandaEETrackingMPCLayer,
    GradMethods,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f'using device: {device}')
    control_mode = 'torque'

    sim_timestep = 0.005
    solve_timestep = 0.01

    # Initialize PyBullet (DIRECT by default; set PYBULLET_GUI=1 for GUI)
    # use_gui = os.environ.get("PYBULLET_GUI", "0") == "1"
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(sim_timestep)
    
    # Load Panda URDF
    urdf_path = "/home/emrea/panda_pytorch/robot_description/panda_with_gripper.urdf"
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE)  # Fix base at ground level
    
    # Add ground plane
    p.loadURDF("plane.urdf", [0, 0, 0])

    # Base initial configuration for the first 7 joints; extra DoF will be zeroed
    base_initial_q7 = [0.0, -0.8, 0.0, -np.pi / 2, 0.0, 0.5, np.pi / 4]
    
    # Instantiate differentiable MPC layer (internally builds dynamics)
    goal_timesteps_abs = torch.tensor([60, 120], dtype=torch.long, device=device)
    layer = PandaEETrackingMPCLayer(
        urdf_path=urdf_path,
        T=10,
        goal_timesteps_abs=goal_timesteps_abs,
        dt=solve_timestep,
        device=device,
        with_gravity=True,
        lqr_iter=1,
        eps=1e-3,
        verbose=0,
    ).to(device)

    n = layer.n_ctrl  # number of actuated DoF

    # Build initial_q of length n
    initial_q = torch.zeros(n, device=device)
    for i in range(min(7, n)):
        initial_q[i] = base_initial_q7[i]

    # Set initial position and disable default motors across actuated joints
    for i in range(n):
        p.resetJointState(robot_id, i, float(initial_q[i].item()))
        # Disable default motor control to allow torque control
        p.setJointMotorControl2(robot_id, i, p.VELOCITY_CONTROL, force=0)
    
    # Enable real-time simulation for better physics
    p.setRealTimeSimulation(0)  # Step simulation manually
    
    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.5]
    )
    # Per-joint effort limits from Pinocchio dynamics
    effort_limits = layer.dynamics.effort_limit.detach().cpu().numpy().tolist()

    # Define two goals to move from point A to point B (batched K=2).
    goal_positions = torch.tensor(
        [
            [0.4,  0.2, 0.5],   # Goal A
            [0.55, -0.2, 0.45], # Goal B
        ], dtype=torch.get_default_dtype(), device=device
    )
    ee_goal_rpy = torch.tensor([3.14, 0.0, 0.0])
    ee_goal_rpy_2 = torch.tensor([1.87, 0.0, 0.0])
    rot = R.from_euler('xyz', ee_goal_rpy.tolist())
    rot2 = R.from_euler('xyz', ee_goal_rpy_2.tolist())
    rot_mat = torch.tensor(rot.as_matrix(), dtype=torch.get_default_dtype(), device=device)
    rot_mat_2 = torch.tensor(rot2.as_matrix(), dtype=torch.get_default_dtype(), device=device)
    goal_rotations = torch.stack([rot_mat, rot_mat_2], dim=0)  # [K,3,3]


    # Weights for cost terms
    v_weight = torch.tensor(1e-2, dtype=torch.get_default_dtype(), device=device)
    u_weight = torch.tensor(1e-9, dtype=torch.get_default_dtype(), device=device)
    pos_weight = torch.tensor(4.0, dtype=torch.get_default_dtype(), device=device)
    orient_weight = torch.tensor(2.0, dtype=torch.get_default_dtype(), device=device)

    # Initial state (current robot state)
    current_q = initial_q.clone().detach()
    current_qdot = torch.zeros(n, device=device)
    x_init = torch.cat([current_q, current_qdot]).unsqueeze(0)  # [B=1, n_state]
    
    # time this mpc solve
    # MPC control loop
    max_steps = int(os.environ.get("MPC_STEPS", "2000"))
    for step in range(max_steps):
        s = time.monotonic()

        # Prepare batched goals for the layer: [B=1, K, ...]
        gp_BK3 = goal_positions.unsqueeze(0)
        gr_BK33 = goal_rotations.unsqueeze(0)

        # Solve MPC via differentiable layer
        x_mpc, u_mpc, obj = layer(
            x_init,
            gp_BK3,
            gr_BK33,
            pos_weight,
            orient_weight,
            v_weight,
            u_weight,
        )

        print(f"elapsed: {time.monotonic() - s}")
        if control_mode == "torque":
            u_cmd = u_mpc[0, 0].detach().cpu().numpy()
            for i in range(n):
                p.setJointMotorControl2(
                    robot_id,
                    i,
                    controlMode=p.TORQUE_CONTROL,
                    force=float(u_cmd[i])
                )
        else:
            x_cmd = x_mpc[4,0].detach().cpu().numpy()
            for i in range(n):
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=float(x_cmd[i].item()),
                    positionGain=0.05,
                    velocityGain=1.0,
                    force=float(effort_limits[i])
                )
        
        # Step simulation
        n_substeps = int(solve_timestep / sim_timestep)
        for i in range(n_substeps):
            p.stepSimulation()
            time.sleep(sim_timestep)            
        
        # Get new state
        for i in range(n):
            current_q[i] = p.getJointState(robot_id, i)[0]
            current_qdot[i] = p.getJointState(robot_id, i)[1]
        
        x_init = torch.cat([current_q, current_qdot]).unsqueeze(0)
        
    
    p.disconnect()

if __name__ == "__main__":
    main()
