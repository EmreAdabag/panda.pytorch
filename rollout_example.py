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
    PinocchioPandaDynamics,
    make_build_ee_tracking_cost,
    MPC,
    QuadCost,
    GradMethods,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f'using device: {device}')

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
    
    # Create Pinocchio-based dynamics model with analytic Jacobians
    dynamics = PinocchioPandaDynamics(
        urdf_path=urdf_path,
        dt=solve_timestep,
        device=device,
        with_gravity=True,
    ).to(device)
    

    n = dynamics.nv  # number of actuated DoF

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
    effort_limits = dynamics.effort_limit.detach().cpu().numpy().tolist()

    # MPC parameters
    n_state = dynamics.nq + dynamics.nv
    n_ctrl = dynamics.nv
    T = 30        # Horizon length (reduced for quick test)
    
    mpc = MPC(
        n_state=n_state,
        n_ctrl=n_ctrl,
        T=T,
        u_lower=None,
        u_upper=None,
        lqr_iter=1,
        verbose=0,
        grad_method=GradMethods.ANALYTIC,
        exit_unconverged=False,  # Don't crash on convergence issues
        detach_unconverged=False,
        eps=1e-3,
        n_batch=1
    )
    
    # End-effector tracking setup (use frame at the hand)
    ee_frame_name = "panda_hand"
    ee_frame_id = dynamics.model.getFrameId(ee_frame_name)
    ee_goal = torch.tensor([0.4, 0.2, 0.5], dtype=torch.get_default_dtype(), device=device)

    ee_goal_rpy = [3.14, 0, 0.]  # roll, pitch, yaw
    # Convert RPY -> Rotation matrix
    rot = R.from_euler('xyz', ee_goal_rpy)   # convention: roll=x, pitch=y, yaw=z
    ee_goal_R = torch.tensor(
        rot.as_matrix(),
        dtype=torch.get_default_dtype(),
        device=device
    )


    # Weights for cost terms
    v_weight = 1e-2
    u_weight = 1e-9
    pos_weight = 5.0
    orient_weight = 1.0
    cost_builder = make_build_ee_tracking_cost(dynamics, ee_frame_id, n_state, n_ctrl)

    # Initial state (current robot state)
    current_q = initial_q.clone().detach()
    current_qdot = torch.zeros(n_ctrl, device=device)
    x_init = torch.cat([current_q, current_qdot]).unsqueeze(0)  # Add batch dimension
    
    # Control loop
    
    # time this mpc solve
    # MPC control loop
    max_steps = int(os.environ.get("MPC_STEPS", "500"))
    for step in range(max_steps):
        s = time.monotonic()

        cost = cost_builder(
            q=current_q,
            ee_goal_pos=ee_goal,
            ee_goal_R=ee_goal_R,
            pos_weight=pos_weight,
            orient_weight=orient_weight,
            v_weight=v_weight,
            u_weight=u_weight,
        )

        # Solve MPC problem and apply next position target
        x_mpc, u_mpc, obj = mpc(x_init, cost, dynamics)
        x_cmd = x_mpc[2,0].detach().cpu().numpy()

        print(f"elapsed: {time.monotonic() - s}")
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
            
        
        # Get new state
        for i in range(n):
            current_q[i] = p.getJointState(robot_id, i)[0]
            current_qdot[i] = p.getJointState(robot_id, i)[1]
        
        x_init = torch.cat([current_q, current_qdot]).unsqueeze(0)
        
    
    p.disconnect()

if __name__ == "__main__":
    main()
