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
import importlib
import importlib.util

# Add paths
sys.path.insert(0, 'mpc.pytorch')
from mpc.mpc import MPC, QuadCost, GradMethods

class PinocchioPandaDynamics(torch.nn.Module):
    """Pinocchio-based dynamics for fixed-base Panda with analytic Jacobians.

    Uses ABA forward dynamics and computeABADerivatives for exact partials.
    Discretization: semi-implicit Euler
        v' = v + dt * a(q, v, u)
        q' = integrate(q, dt * v')   (approx Jacobians treat integrate as q + dt*v')

    Notes
    - Pinocchio runs on CPU; tensors are copied to CPU for dynamics, results
      are returned on the input device. Batch is processed in a Python loop.
    - For speed: reuse model/data and avoid allocations where possible.
    """
    def __init__(self, urdf_path, dt=0.01, device=None, with_gravity=True):
        super().__init__()
        self.dt = float(dt)
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        self.pin = pin

        # Build model and data
        self.model = self.pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        if with_gravity:
            self.model.gravity.linear = np.array([0.0, 0.0, -9.81])
        else:
            self.model.gravity.linear = np.array([0.0, 0.0, 0.0])

        # Sizes (generalized for any actuated DoF)
        self.nq = self.model.nq
        self.nv = self.model.nv

        # Joint limits and efforts (from Pinocchio model)
        q_lower = torch.tensor(self.model.lowerPositionLimit, dtype=torch.float64)
        q_upper = torch.tensor(self.model.upperPositionLimit, dtype=torch.float64)
        effort_limit = torch.tensor(self.model.effortLimit, dtype=torch.float64)
        self.register_buffer('q_lower', q_lower.to(self.device))
        self.register_buffer('q_upper', q_upper.to(self.device))
        self.register_buffer('effort_limit', effort_limit.to(self.device))

    def _to_numpy(self, t):
        return t.detach().cpu().numpy()

    def _forward_single(self, q_np, v_np, u_np):
        pin = self.pin
        # Compute forward dynamics and semi-implicit integration
        a = pin.aba(self.model, self.data, q_np, v_np, u_np)  # shape (7,)
        v_next = v_np + self.dt * a
        # Simple explicit Euler for configuration update
        q_next = q_np + self.dt * v_next
        return q_next, v_next

    def forward(self, x, u):
        # x: [B, 2*n], u: [B, n]
        B = x.shape[0]
        dev = x.device
        dtype = x.dtype

        n = self.nv
        q = x[:, :n]
        v = x[:, n:]
        # No clamping for minimal implementation

        q_next_list = []
        v_next_list = []
        for i in range(B):
            q_np = self._to_numpy(q[i]).astype(np.float64)
            v_np = self._to_numpy(v[i]).astype(np.float64)
            u_np = self._to_numpy(u[i]).astype(np.float64)
            qn, vn = self._forward_single(q_np, v_np, u_np)
            q_next_list.append(torch.from_numpy(np.asarray(qn)).to(device=dev, dtype=dtype))
            v_next_list.append(torch.from_numpy(np.asarray(vn)).to(device=dev, dtype=dtype))

        q_next = torch.stack(q_next_list, dim=0)
        v_next = torch.stack(v_next_list, dim=0)
        return torch.cat([q_next, v_next], dim=-1)

    @torch.no_grad()
    def grad_input(self, x, u):
        # Returns R, S with shapes [B,2n,2n] and [B,2n,n]
        B = x.shape[0]
        dev = x.device
        dtype = x.dtype
        dt = self.dt

        pin = self.pin
        n = self.nv
        In = torch.eye(n, device=dev, dtype=dtype).expand(B, n, n)

        # Prepare outputs
        R = torch.zeros(B, 2*n, 2*n, device=dev, dtype=dtype)
        S = torch.zeros(B, 2*n, n, device=dev, dtype=dtype)

        q = x[:, :n]
        v = x[:, n:]

        for i in range(B):
            q_np = self._to_numpy(q[i]).astype(np.float64)
            v_np = self._to_numpy(v[i]).astype(np.float64)
            u_np = self._to_numpy(u[i]).astype(np.float64)

            # Compute derivatives of acceleration: dadq, dadv, dadu in R^{n x n}
            d_dq, d_dv, d_du = pin.computeABADerivatives(self.model, self.data, q_np, v_np, u_np)
            # Convert to torch
            dadq = torch.from_numpy(np.asarray(d_dq)).to(device=dev, dtype=dtype)
            dadv = torch.from_numpy(np.asarray(d_dv)).to(device=dev, dtype=dtype)
            dadu = torch.from_numpy(np.asarray(d_du)).to(device=dev, dtype=dtype)

            # Semi-implicit Euler discrete-time Jacobians
            # v' = v + dt*a  => dv'/dq = dt*dadq, dv'/dv = I + dt*dadv, dv'/du = dt*dadu
            R22 = In[i] + dt * dadv
            R21 = dt * dadq
            S_bot = dt * dadu

            # Explicit Euler for position: q' = q + dt*v'
            R11 = In[i] + dt * R21
            R12 = dt * R22
            S_top = dt * S_bot

            # Assemble into R and S
            # Top rows correspond to q'
            R[i, 0:n, 0:n] = R11
            R[i, 0:n, n:2*n] = R12
            # Bottom rows correspond to v'
            R[i, n:2*n, 0:n] = R21
            R[i, n:2*n, n:2*n] = R22

            S[i, 0:n, :] = S_top
            S[i, n:2*n, :] = S_bot

        return R, S


class EETrackingCostFn(torch.autograd.Function):
    """Custom autograd for assembling quadratic cost C, c.

    Provides gradients w.r.t. ee_goal_pos and weights. Gradients for q are
    not provided (treated as constant here). Orientation gradient w.r.t.
    ee_goal_R is not implemented analytically in this version.
    """

    @staticmethod
    def forward(ctx, x_state, ee_goal_pos, ee_goal_R, pos_weight, orient_weight, v_weight, u_weight, dynamics, ee_frame_id, n_state, n_ctrl):
        device = x_state.device
        dtype = x_state.dtype

        # Kinematics/Jacobians via Pinocchio at the given q
        # Use only the configuration part for kinematics
        assert x_state.ndimension() == 1 and x_state.numel() == n_state
        q = x_state[:n_ctrl]
        q_np = q.detach().cpu().numpy().astype(np.float64)
        dynamics.pin.forwardKinematics(dynamics.model, dynamics.data, q_np)
        pin.updateFramePlacements(dynamics.model, dynamics.data)
        fk = dynamics.data.oMf[ee_frame_id]
        p_cur = torch.from_numpy(np.asarray(fk.translation)).to(device=device, dtype=dtype)
        R_cur = np.asarray(fk.rotation)

        e_pos = p_cur - ee_goal_pos
        R_err = ee_goal_R.detach().cpu().numpy() @ R_cur.T
        e_rot_np = pin.log3(R_err)
        e_rot = torch.from_numpy(np.asarray(e_rot_np)).to(device=device, dtype=dtype)

        J6 = pin.computeFrameJacobian(
            dynamics.model,
            dynamics.data,
            q_np,
            ee_frame_id,
            pin.ReferenceFrame.WORLD,
        )
        J_pos = torch.from_numpy(np.asarray(J6[:3, :])).to(device=device, dtype=dtype)
        J_rot = torch.from_numpy(np.asarray(J6[3:6, :])).to(device=device, dtype=dtype)

        # Quadratic terms
        n = n_ctrl
        Qq = pos_weight * (J_pos.T @ J_pos) + orient_weight * (J_rot.T @ J_rot)
        pq = pos_weight * (J_pos.T @ e_pos) - orient_weight * (J_rot.T @ e_rot)
        pq = pq - Qq @ q

        C = torch.zeros(n_state + n_ctrl, n_state + n_ctrl, device=device, dtype=dtype)
        C[0:n, 0:n] = Qq
        C[n:2*n, n:2*n] = v_weight * torch.eye(n, device=device, dtype=dtype)
        C[2*n:, 2*n:] = u_weight * torch.eye(n_ctrl, device=device, dtype=dtype)

        c = torch.zeros(n_state + n_ctrl, device=device, dtype=dtype)
        c[0:n] = pq

        # Save for backward
        ctx.save_for_backward(J_pos, J_rot, e_pos, e_rot, q, ee_goal_R,
                              torch.tensor(n_state), torch.tensor(n_ctrl),
                              pos_weight, orient_weight, v_weight, u_weight)
        ctx.dynamics = dynamics
        ctx.ee_frame_id = ee_frame_id
        return C, c

    @staticmethod
    def backward(ctx, gC, gc):
        J_pos, J_rot, e_pos, e_rot, q, ee_goal_R, n_state_t, n_ctrl_t, pos_w, ori_w, v_w, u_w = ctx.saved_tensors
        n_state = int(n_state_t.item())
        n_ctrl = int(n_ctrl_t.item())

        # Upstream grads slices
        n = n_ctrl
        gCqq = gC[0:n, 0:n]
        gCvv = gC[n:2*n, n:2*n]
        gCuu = gC[2*n:, 2*n:]
        gc_q = gc[0:n]

        # Gradients init
        g_q = None  # no gradient for q

        # ee_goal_pos gradient: d pq / d goal_pos = -pos_w * J_pos^T applied to gc_q
        gee_pos = -(pos_w.item()) * (J_pos @ gc_q)

        # ee_goal_R gradient via SO(3) log residual
        # Upstream gradient on e_rot from c_q term: c_q includes (-ori_w * J_rot^T e_rot)
        g_e = -(ori_w.item()) * (J_rot @ gc_q)  # shape (3,)

        def hat(v):
            vx, vy, vz = v[0], v[1], v[2]
            H = torch.zeros(3, 3, dtype=v.dtype, device=v.device)
            H[0,1] = -vz; H[0,2] =  vy
            H[1,0] =  vz; H[1,2] = -vx
            H[2,0] = -vy; H[2,1] =  vx
            return H

        def so3_right_jacobian_inv(phi):
            I = torch.eye(3, dtype=phi.dtype, device=phi.device)
            theta = torch.norm(phi)
            eps = torch.tensor(1e-8, dtype=phi.dtype, device=phi.device)
            H = hat(phi)
            theta_clamped = torch.clamp(theta, min=eps)
            # stable b term
            half = -0.5
            if theta.item() < 1e-5:
                b = 1.0/12.0
            else:
                b = (1.0/(theta_clamped*theta_clamped) - (1.0+torch.cos(theta_clamped))/(2.0*theta_clamped*torch.sin(theta_clamped)))
            return I + half*H + b*(H @ H)

        Jr_inv = so3_right_jacobian_inv(e_rot)
        # A = (Jr_inv)^T * g_e
        A = Jr_inv.t() @ g_e
        # Gee_R = 0.5 * R_des * hat(R_des^T * A)
        R_des = ee_goal_R
        Gee_R = 0.5 * (R_des @ hat(R_des.t() @ A))

        # pos_weight gradient
        JpTJp = J_pos.T @ J_pos
        termC_pos = (gCqq * JpTJp).sum()
        termc_pos = gc_q @ (J_pos.T @ e_pos - JpTJp @ q)
        g_pos_w = termC_pos + termc_pos

        # orient_weight gradient
        JrTJr = J_rot.T @ J_rot
        termC_ori = (gCqq * JrTJr).sum()
        termc_ori = gc_q @ (- J_rot.T @ e_rot - JrTJr @ q)
        g_ori_w = termC_ori + termc_ori

        # v_weight gradient: only on diagonal block
        g_v_w = torch.trace(gCvv)

        # u_weight gradient: only on diagonal block
        g_u_w = torch.trace(gCuu)

        # None for dynamics-related saved constants
        return (
            g_q,
            gee_pos,
            Gee_R,
            g_pos_w,
            g_ori_w,
            g_v_w,
            g_u_w,
            None,
            None,
            None,
            None,
        )


# def build_ee_tracking_cost_runtime(
#     q: torch.Tensor,
#     T: torch.Tensor,
#     goal_timesteps: torch.Tensor,
#     goal_positions: torch.Tensor,
#     goal_rotations: torch.Tensor,
#     dynamics: PinocchioPandaDynamics,
#     pos_weight: torch.Tensor,
#     orient_weight: torch.Tensor,
#     v_weight: torch.Tensor,
#     u_weight: torch.Tensor,
# ):
#     """Assemble a time-varying QuadCost for multiple goals provided at runtime.

#     All inputs must be tensors. Fails with assertions if shapes/dtypes mismatch.

#     Args (tensors):
#       q: [n_ctrl]
#       T: scalar long tensor (horizon)
#       goal_timesteps: [K] long tensor; horizon-relative indices, can be > T-1
#       goal_positions: [K, 3]
#       goal_rotations: [K, 3, 3]
#       pos_weight, orient_weight, v_weight, u_weight: scalar tensors
#     """
#     assert isinstance(q, torch.Tensor) and q.ndimension() == 1
#     assert isinstance(T, torch.Tensor) and T.numel() == 1
#     assert isinstance(goal_timesteps, torch.Tensor) and goal_timesteps.ndimension() == 1
#     assert isinstance(goal_positions, torch.Tensor) and goal_positions.ndimension() == 2 and goal_positions.size(1) == 3
#     assert isinstance(goal_rotations, torch.Tensor) and goal_rotations.ndimension() == 3 and goal_rotations.size(1) == 3 and goal_rotations.size(2) == 3
#     assert goal_timesteps.size(0) == goal_positions.size(0) == goal_rotations.size(0)

#     device = q.device
#     dtype = q.dtype
#     T_int = int(T.item())
#     K = goal_timesteps.size(0)

#     # Infer fixed parameters from dynamics
#     ee_frame_id = dynamics.model.getFrameId("panda_hand")
#     n_ctrl = dynamics.nv
#     n_state = dynamics.nq + dynamics.nv

#     # Prepare weights on correct device/dtype
#     pw = pos_weight.to(device=device, dtype=dtype)
#     ow = orient_weight.to(device=device, dtype=dtype)
#     vw = v_weight.to(device=device, dtype=dtype)
#     uw = u_weight.to(device=device, dtype=dtype)

#     # Sort goals by timestep ascending
#     sort_idx = torch.argsort(goal_timesteps)
#     ts_sorted = goal_timesteps[sort_idx]
#     gp_sorted = goal_positions[sort_idx].to(device=device, dtype=dtype)
#     gr_sorted = goal_rotations[sort_idx].to(device=device, dtype=dtype)

#     # Build sequences without in-place writes to preserve autograd graph
#     n_tau = n_state + n_ctrl
#     C_list = []
#     c_list = []

#     # Map each horizon timestep t to the "next goal" and build via EETrackingCostFn
#     for t in range(T_int):
#         idx = None
#         for k in range(K):
#             if t <= int(ts_sorted[k].item()):
#                 idx = k
#                 break
#         if idx is None:
#             idx = K - 1

#         C_t, c_t = EETrackingCostFn.apply(
#             q,
#             gp_sorted[idx],
#             gr_sorted[idx],
#             pw,
#             ow,
#             vw,
#             uw,
#             dynamics,
#             ee_frame_id,
#             n_state,
#             n_ctrl,
#         )
#         C_list.append(C_t)
#         c_list.append(c_t)

#     C_seq = torch.stack(C_list, dim=0)
#     c_seq = torch.stack(c_list, dim=0)
#     return QuadCost(C_seq, c_seq)


def build_ee_tracking_cost_batched_timevarying(
    x_batch: torch.Tensor,
    T: int,
    goal_timesteps: torch.Tensor,
    goal_positions: torch.Tensor,
    goal_rotations: torch.Tensor,
    dynamics: PinocchioPandaDynamics,
    pos_weight: torch.Tensor,
    orient_weight: torch.Tensor,
    v_weight: torch.Tensor,
    u_weight: torch.Tensor,
):
    """Batched, time-varying QuadCost using a shared goal schedule across batch.

    For each time t in [0, T-1], selects the "next goal" index from
    `goal_timesteps` (following build_ee_tracking_cost_runtime), then assembles
    per-batch quadratic costs via EETrackingCostFn.

    Args:
      q_batch: [B, n_ctrl]
      T: int horizon length
      goal_timesteps: [K] long tensor (horizon-relative, can exceed T-1)
      goal_positions: [B, K, 3]
      goal_rotations: [B, K, 3, 3]
      pos_weight, orient_weight, v_weight, u_weight: scalar or [B]
    Returns QuadCost with C: [T, B, n_tau, n_tau], c: [T, B, n_tau]
    """
    assert x_batch.ndimension() == 2, "x_batch must be [B, n_state]"
    B, n_state = x_batch.shape
    device, dtype = x_batch.device, x_batch.dtype

    ee_frame_id = dynamics.model.getFrameId("panda_hand")
    n_ctrl = dynamics.nv
    assert n_state == dynamics.nq + dynamics.nv, "x_batch second dim must match n_state"

    # Require scalar weights for simplicity
    assert pos_weight.ndimension() == 0
    assert orient_weight.ndimension() == 0
    assert v_weight.ndimension() == 0
    assert u_weight.ndimension() == 0

    K = goal_timesteps.size(0)
    assert goal_positions.shape == (B, K, 3), "goal_positions must be [B,K,3]"
    assert goal_rotations.shape == (B, K, 3, 3), "goal_rotations must be [B,K,3,3]"

    # Sort goal schedule indices once (shared across batch)
    sort_idx = torch.argsort(goal_timesteps)
    ts_sorted = goal_timesteps[sort_idx]

    C_seq = []
    c_seq = []
    for t in range(T):
        # find next goal index according to runtime builder's rule
        idx = None
        for k in range(K):
            if t <= int(ts_sorted[k].item()):
                idx = sort_idx[k]
                break
        if idx is None:
            idx = sort_idx[-1]

        Ct_b = []
        ct_b = []
        for b in range(B):
            Cb, cb = EETrackingCostFn.apply(
                x_batch[b],
                goal_positions[b, idx],
                goal_rotations[b, idx],
                pos_weight,
                orient_weight,
                v_weight,
                u_weight,
                dynamics,
                ee_frame_id,
                n_state,
                n_ctrl,
            )
            Ct_b.append(Cb)
            ct_b.append(cb)

        C_seq.append(torch.stack(Ct_b, dim=0))  # [B, n_tau, n_tau]
        c_seq.append(torch.stack(ct_b, dim=0))  # [B, n_tau]

    C_seq = torch.stack(C_seq, dim=0)  # [T, B, n_tau, n_tau]
    c_seq = torch.stack(c_seq, dim=0)  # [T, B, n_tau]
    return QuadCost(C_seq, c_seq)


class PandaEETrackingMPCLayer(torch.nn.Module):
    """Differentiable MPC layer for Panda EE tracking with batch support.

    - Bakes in `dynamics` and horizon `T` at init time.
    - Accepts per-batch goal positions/rotations and scalar weights at runtime.
    - Maintains an absolute goal schedule (timesteps) internally and advances
      it per forward call to build a time-varying cost.
    - Builds a quadratic cost via `EETrackingCostFn` and solves MPC.

    Forward inputs
      x_init: [B, n_state] current state (q concatenated with dq)
      goal_positions: [B, K, 3]
      goal_rotations: [B, K, 3, 3]
      pos_weight, orient_weight, v_weight, u_weight: scalar tensors

    Returns
      x_traj: [T, B, n_state]
      u_traj: [T, B, n_ctrl]
      costs: [B]
    """
    def __init__(
        self,
        urdf_path: str,
        T: int,
        goal_timesteps_abs: torch.Tensor,
        dt: float = 0.01,
        device=None,
        with_gravity: bool = True,
        lqr_iter: int = 1,
        eps: float = 1e-3,
        verbose: int = 0,
    ):
        super().__init__()
        # Initialize internal dynamics
        self.dynamics = PinocchioPandaDynamics(
            urdf_path=urdf_path, dt=dt, device=device, with_gravity=with_gravity
        )
        self.T = int(T)
        self.n_state = self.dynamics.nq + self.dynamics.nv
        self.n_ctrl = self.dynamics.nv

        # Absolute goal schedule and internal rollout step counter
        assert isinstance(goal_timesteps_abs, torch.Tensor) and goal_timesteps_abs.ndimension() == 1
        self.register_buffer('goal_timesteps_abs', goal_timesteps_abs.clone().long())
        self.register_buffer('rollout_step', torch.zeros((), dtype=torch.long))

        # effort_max = self.dynamics.effort_limit.abs().unsqueeze(0).unsqueeze(0).repeat(T, 1, 1).to(dtype=torch.float)
        # effort_min = -1. * effort_max
        effort_max = effort_min = None

        self.mpc = MPC(
            n_state=self.n_state,
            n_ctrl=self.n_ctrl,
            T=self.T,
            u_lower=effort_min,
            u_upper=effort_max,
            lqr_iter=lqr_iter, #lqr_iter,
            grad_method=GradMethods.ANALYTIC,
            verbose=verbose,
            eps=eps,
            n_batch=None,  # infer from cost
            exit_unconverged=False,
            detach_unconverged=False,
        )

    def forward(
        self,
        x_init: torch.Tensor,
        goal_positions: torch.Tensor,
        goal_rotations: torch.Tensor,
        pos_weight: torch.Tensor,
        orient_weight: torch.Tensor,
        v_weight: torch.Tensor,
        u_weight: torch.Tensor,
    ):
        assert x_init.ndimension() == 2 and x_init.size(1) == self.n_state
        B = x_init.size(0)
        # Require scalar weights for simplicity
        assert pos_weight.ndimension() == 0
        assert orient_weight.ndimension() == 0
        assert v_weight.ndimension() == 0
        assert u_weight.ndimension() == 0

        # Compute horizon-relative timesteps from absolute schedule and
        # current internal rollout step.
        rel_ts = (self.goal_timesteps_abs - self.rollout_step).to(device=x_init.device)

        cost = build_ee_tracking_cost_batched_timevarying(
            x_batch=x_init,
            T=self.T,
            goal_timesteps=rel_ts,
            goal_positions=goal_positions,
            goal_rotations=goal_rotations,
            dynamics=self.dynamics,
            pos_weight=pos_weight,
            orient_weight=orient_weight,
            v_weight=v_weight,
            u_weight=u_weight,
        )

        x_traj, u_traj, costs = self.mpc(x_init, cost, self.dynamics)
        # Advance internal rollout step
        self.rollout_step += 1
        # Detach costs to avoid backprop through extra LQRStep outputs
        return x_traj, u_traj, costs.detach()

    def reset_schedule(self, step: int = 0):
        """Reset internal rollout step (e.g., at episode start)."""
        self.rollout_step = torch.tensor(int(step), dtype=torch.long, device=self.rollout_step.device)
