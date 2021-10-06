from collections import namedtuple

import jax.numpy as jnp
import jax.scipy.linalg


class StackedSSM:
    def __init__(self, processes) -> None:
        self.processes = tuple(processes)
        self._dims = tuple((p.state_dimension for p in self.processes))

    @property
    def state_dimension(self):
        return sum(self._dims)

    @property
    def preconditioned_discretize(self):
        As, Qs = [], []
        for p in self.processes:
            A, SQ = p.preconditioned_discretize
            As.append(A)
            Qs.append(SQ)

        A = jax.scipy.linalg.block_diag(*As)
        Q = jax.scipy.linalg.block_diag(*Qs)
        return A, Q

    def non_preconditioned_discretize(self, dt):
        As, Qs = [], []
        for p in self.processes:
            A, SQ = p.non_preconditioned_discretize(dt)
            As.append(A)
            Qs.append(SQ)

        A = jax.scipy.linalg.block_diag(*As)
        Q = jax.scipy.linalg.block_diag(*Qs)
        return A, Q

    def nordsieck_preconditioner(self, dt):
        Ps, P_invs = [], []
        for p in self.processes:
            prec, prec_inv = p.nordsieck_preconditioner(dt)
            Ps.append(prec)
            P_invs.append(prec_inv)

        P = jax.scipy.linalg.block_diag(*Ps)
        P_inv = jax.scipy.linalg.block_diag(*P_invs)
        return P, P_inv

    def projection_matrix(
        self, derivative_to_project_onto, process_to_project_onto=None
    ):
        if process_to_project_onto is None:
            return jax.scipy.linalg.block_diag(
                *[
                    p.projection_matrix(derivative_to_project_onto)
                    for p in self.processes
                ]
            )

        assert isinstance(process_to_project_onto, int)
        proj_to_proc = self.projection_to_process(process_to_project_onto)
        proj_to_deriv = self.processes[process_to_project_onto].projection_matrix(
            derivative_to_project_onto
        )
        return proj_to_deriv @ proj_to_proc

    def projection_to_process(self, process_to_project_onto: int):
        start = (
            sum(self._dims[0:process_to_project_onto])
            if process_to_project_onto > 0
            else 0
        )
        stop = (
            start + self._dims[process_to_project_onto]
            if process_to_project_onto < len(self.processes)
            else None
        )
        return jnp.eye(self.state_dimension)[start:stop, :]
