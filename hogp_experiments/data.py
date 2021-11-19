import math
import numpy as np
import torch

from botorch.acquisition.objective import GenericMCObjective

# from utils import PRE, cfun


def prepare_data(problem, s_size, t_size, device, dtype):
    if problem == "environmental":
        print("---- Running the environmental problem with ", s_size, t_size, " ----")
        # X = [M, D, L, tau]
        bounds = torch.tensor(
            [[7.0, 0.02, 0.01, 30.010], [13.0, 0.12, 3.00, 30.295]],
            device=device,
            dtype=dtype,
        )

        M0 = torch.tensor(10.0, device=device, dtype=dtype)
        D0 = torch.tensor(0.07, device=device, dtype=dtype)
        L0 = torch.tensor(1.505, device=device, dtype=dtype)
        tau0 = torch.tensor(30.1525, device=device, dtype=dtype)

        # we can vectorize everything, no need for loops
        if s_size == 3:
            S = torch.tensor([0.0, 1.0, 2.5], device=device, dtype=dtype)
        else:
            S = torch.linspace(0.0, 2.5, s_size, device=device, dtype=dtype)
        if t_size == 4:
            T = torch.tensor([15.0, 30.0, 45.0, 60.0], device=device, dtype=dtype)
        else:
            T = torch.linspace(15.0, 60.0, t_size, device=device, dtype=dtype)

        Sgrid, Tgrid = torch.meshgrid(S, T)

        # X = [M, D, L, tau]
        def c_batched(X, k=None):
            return torch.stack([env_cfun(Sgrid, Tgrid, *x) for x in X])

        c_true = env_cfun(Sgrid, Tgrid, M0, D0, L0, tau0)

        def neq_sum_quared_diff(samples):
            # unsqueeze
            if samples.shape[-1] == (s_size * t_size):
                samples = samples.unsqueeze(-1).reshape(
                    *samples.shape[:-1], s_size, t_size
                )

            sq_diffs = (samples - c_true).pow(2)
            return sq_diffs.sum(dim=(-1, -2)).mul(-1.0)

        objective = GenericMCObjective(neq_sum_quared_diff)
        num_samples = 32

    elif problem == "precursor":
        print("---- Running the precursor problem ----")
        bounds = torch.tensor(
            [[-6.0, -6.0, -6.0], [6.0, 6.0, 6.0]], device=device, dtype=dtype,
        )

        tf = 10
        dt = 0.01

        def simulate(theta, tf=50, dt=0.01):
            nsteps = int(tf / dt)
            pre = PRE(tf, nsteps, theta)
            u, _ = pre.solve()
            return u

        def c_batched(X, k=None):
            return (
                torch.stack([torch.tensor(simulate(x.cpu(), tf=tf, dt=dt)) for x in X])
                .to(X.device)
                .type(dtype)
            )

        def objective_fn(samples):
            shapes = [int(tf / dt), 3]
            if samples.shape[-1] == (shapes[0] * shapes[1]):
                samples = samples.unsqueeze(-1).reshape(*samples.shape[:-1], *shapes)

            # return -torch.max(samples[..., -1], dim=-1)[0]
            # differentiable version
            lse = torch.logsumexp(samples[..., -1], dim=-1).log()
            return -lse

        objective = GenericMCObjective(objective_fn)

        num_samples = 4
    elif problem == "maveric1":
        # from ax.fb.utils.storage.manifold import AEManifoldUseCase
        # from ax.fb.utils.storage.manifold_torch import AEManifoldTorchClient
        import sys

        sys.path.append("../../fbc-maveric-research/")
        from fbc.maveric.simulation_data.simulated_rsrp import SimulatedRSRP

        min_Tx_power_dBm, max_Tx_power_dBm = 30, 50

        # loads the 11 powermap files that are typically mounted in
        # "/mnt/shared/yuchenq/power_maps/*.npz"
        # and does the same type of powermap construction as in
        # fbc.maveric.simulation_data.simulated_rsrp.construct_from_npz_files

        import glob
        import json

        powermaps = glob.glob("../data/powermatrixDT*.json")
        all_pmap_dicts = []
        for pmap_loc in powermaps:
            with open(pmap_loc, "r") as f:
                all_pmap_dicts.append(json.load(f))

        downtilts_maps = {}
        for i in range(11):
            downtilts_maps[float(i)] = SimulatedRSRP.build_single_powermap(
                all_pmap_dicts[i]
            )

        simulated_rsrp = SimulatedRSRP(
            downtilts_maps=downtilts_maps,
            min_TX_power_dBm=min_Tx_power_dBm,
            max_TX_power_dBm=max_Tx_power_dBm,
        )

        def simulate(theta):
            theta = theta.cpu().detach().numpy()
            downtilts = theta[:15].astype(int)
            tx_pwrs = theta[15:]
            (
                rsrp_powermap,
                interference_powermap,
                _,
            ) = simulated_rsrp.get_RSRP_and_interference_powermap((downtilts, tx_pwrs))
            # return torch.stack(
            #     (torch.tensor(rsrp_powermap), torch.tensor(interference_powermap))
            # )
            highd_res = torch.stack(
                (torch.tensor(rsrp_powermap), torch.tensor(interference_powermap))
            )
            return torch.nn.functional.interpolate(highd_res.unsqueeze(0), (50, 50))[0]

        def c_batched(X, k=None):
            return torch.stack([simulate(x) for x in X]).to(X.device).type(dtype)

        lower_bounds = torch.tensor([0.0] * 15 + [min_Tx_power_dBm] * 15)
        upper_bounds = torch.tensor([10.0] * 15 + [max_Tx_power_dBm] * 15)
        bounds = torch.stack((lower_bounds, upper_bounds))

        # this is a crude, but differentiable version of
        # fbc/maveric/coverage_capacity_optimization/problem_formulation.py
        # TODO: use that version
        def construct_both_objectives(samples):
            rsrp_map = samples[..., 0, :, :]
            interference_map = samples[..., 1, :, :]

            weak_coverage_threshold = -80.0
            over_coverage_threshold = 6.0
            f_weak_coverage = torch.sigmoid(weak_coverage_threshold - rsrp_map).sum(
                dim=(-1, -2)
            )
            size = np.prod(rsrp_map.shape[-2:])

            # over_coverage_area = (rsrp_map >= weak_coverage_threshold) & (
            #    interference_map + over_coverage_threshold > rsrp_map
            # )
            rsrp_gt_threshold = torch.sigmoid(rsrp_map - weak_coverage_threshold)
            if_gt_threshold = torch.sigmoid(
                (interference_map + over_coverage_threshold) - rsrp_map
            )
            over_coverage_area = rsrp_gt_threshold * if_gt_threshold

            over_coverage_map = (
                interference_map * over_coverage_area
                + over_coverage_threshold
                - rsrp_map * over_coverage_area
            )
            # over_coverage_map = (
            #     interference_map[over_coverage_area]
            #     + over_coverage_threshold
            #     - rsrp_map[over_coverage_area]
            # )
            g_weak_coverage = torch.sigmoid(over_coverage_map).sum(dim=(-1, -2))
            return f_weak_coverage, g_weak_coverage

        # this is a scalarization for now
        def objective_fn(samples):
            weight = 0.25
            f_weak_coverage, g_weak_coverage = construct_both_objectives(samples)
            return weight * f_weak_coverage + (1.0 - weight) * g_weak_coverage

        objective = GenericMCObjective(objective_fn)

        num_samples = 4

    elif problem == "pde":
        print("Running the brusselator")

        def cfun(tensor, k=None):
            from pde import PDE, FieldCollection, ScalarField, UnitGrid

            a = tensor[0].item()
            b = tensor[1].item()
            d0 = tensor[2].item()
            d1 = tensor[3].item()

            eq = PDE(
                {
                    "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
                    "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
                }
            )

            # initialize state
            grid = UnitGrid([64, 64])
            u = ScalarField(grid, a, label="Field $u$")
            v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$", seed=10)
            state = FieldCollection([u, v])

            sol = eq.solve(state, t_range=20, dt=1e-3)
            sol_tensor = torch.stack(
                (torch.from_numpy(sol[0].data), torch.from_numpy(sol[1].data))
            )
            sol_tensor[~torch.isfinite(sol_tensor)] = 1e5 * torch.randn_like(
                sol_tensor[~torch.isfinite(sol_tensor)]
            )
            return sol_tensor

        def c_batched(X, k):
            return torch.stack([cfun(x) for x in X]).to(X.device).type(dtype)

        def objective_fn(samples, use_var=True):
            # we want to minimize the variance across the boundaries
            sz = samples.shape[-1]
            weighting = (
                torch.ones(2, sz, sz, device=samples.device, dtype=samples.dtype) / 10
            )
            weighting[:, [0, 1, -2, -1], :] = 1.0
            weighting[:, :, [0, 1, -2, -1]] = 1.0

            weighted_samples = weighting * samples
            if use_var:
                return -weighted_samples.var(dim=(-1, -2, -3))
            else:
                return -weighted_samples.sum(dim=(-1, -2, -3))

        objective = GenericMCObjective(objective_fn)

        num_samples = 32
        lb = torch.tensor([0.1, 0.1, 0.01, 0.01])
        ub = torch.tensor([5.0, 5.0, 5.0, 5.0])
        bounds = torch.stack((lb, ub))
    elif problem == "optics":
        from gym_interf import InterfEnv

        gym_dict = {}

        def metric(image):
            xx, yy = torch.meshgrid(
                torch.arange(64, dtype=image.dtype, device=image.device) / 64,
                torch.arange(64, dtype=image.dtype, device=image.device) / 64,
            )
            intens = (-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2).exp() * image
            ivec = intens.sum((-1, -2))
            smax = torch.logsumexp(ivec, -1)
            smin = -torch.logsumexp(-ivec, -1)
            v = (smax - smin) / (smax + smin)
            return v.log()  # - (1 - v).log()

        def cfun(x, k):
            # if k not in gym_dict.keys():
            gym_dict[k] = InterfEnv()
            # gym_dict[k].reset(actions=x[4:].cpu().detach().numpy())
            gym_dict[k].reset(actions=(1e-4, 1e-4, 1e-4, 1e-4))

            action = x[:4].cpu().detach().numpy()
            state = gym_dict[k].step(action)
            return torch.tensor(state[0])

        def c_batched(X, k):
            return torch.stack([cfun(x, k) for x in X]).to(X.device).type(dtype)

        upper_bounds = torch.ones(4, device=device, dtype=dtype)
        lower_bounds = -1.0 * upper_bounds
        bounds = torch.stack((lower_bounds, upper_bounds))

        objective = GenericMCObjective(metric)

        num_samples = 4

    return c_batched, objective, bounds, num_samples


def env_cfun(s, t, M, D, L, tau):
    c1 = M / torch.sqrt(4 * math.pi * D * t)
    exp1 = torch.exp(-(s ** 2) / 4 / D / t)
    term1 = c1 * exp1
    c2 = M / torch.sqrt(4 * math.pi * D * (t - tau))
    exp2 = torch.exp(-((s - L) ** 2) / 4 / D / (t - tau))
    term2 = c2 * exp2
    term2[torch.isnan(term2)] = 0.0
    return term1 + term2


class PRE:
    def __init__(self, tf, nsteps, u_init, alp=0.01, ome=2 * np.pi, lam=0.1, bet=0.1):
        self.alp = alp
        self.ome = ome
        self.lam = lam
        self.bet = bet
        self.tf = tf
        self.nsteps = nsteps
        self.u_init = u_init

    def RHS(self, u, t):
        x1, x2, x3 = u
        f1 = (
            self.alp * x1
            + self.ome * x2
            + self.alp * x1 ** 2
            + 2 * self.ome * x1 * x2
            + x3 ** 2
        )
        f2 = -self.ome * x1 + self.alp * x2 - self.ome * x1 ** 2 + 2 * self.alp * x1 * x2
        f3 = -self.lam * x3 - (self.lam + self.bet) * x1 * x3
        f = [f1, f2, f3]
        return f

    def solve(self):
        time = np.linspace(0, self.tf, self.nsteps + 1)
        solver = ODESolver(self.RHS)
        solver.set_ics(self.u_init)
        u, t = solver.solve(time)
        return u, t


class ODESolver:
    def __init__(self, f):
        self.f = lambda u, t: np.asarray(f(u, t), float)

    def set_ics(self, U0):
        U0 = np.asarray(U0)
        self.neq = U0.size
        self.U0 = U0

    def advance(self):
        u, f, k, t = self.u, self.f, self.k, self.t
        dt = t[k + 1] - t[k]
        K1 = dt * f(u[k], t[k])
        K2 = dt * f(u[k] + 0.5 * K1, t[k] + 0.5 * dt)
        K3 = dt * f(u[k] + 0.5 * K2, t[k] + 0.5 * dt)
        K4 = dt * f(u[k] + K3, t[k] + dt)
        u_new = u[k] + (1 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)
        return u_new

    def solve(self, time):
        self.t = np.asarray(time)
        n = self.t.size
        self.u = np.zeros((n, self.neq))
        self.u[0] = self.U0
        for k in range(n - 1):
            self.k = k
            self.u[k + 1] = self.advance()
        return self.u[: k + 2], self.t[: k + 2]
