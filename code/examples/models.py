import numpy as np
from neuromancer.psl.signals import step, sines, periodic, noise, walk
from neuromancer.psl.base import ODE_NonAutonomous as ODE
from neuromancer.psl.base import cast_backend
import inspect, sys

class TwoTank(ODE):
    """
    Two Tank model. with transport delay
    """
    @property
    def params(self):
        variables = {'x0': [0., 0.],}
        constants = {'ts': 10.0}
        parameters = {'F1': 1,  
                      'F2': 1,
                      'k11': 0.015,
                      'k22': 0.015,
                      }
        meta = {}
        return variables, constants, parameters, meta

    @property
    def umin(self):
        return np.array([0.0], dtype=np.float32)

    @property
    def umax(self):
        """
        Note that although the theoretical upper bound is 1.0,
        this results in numerical instability in the integration.
        """
        return np.array([0.03], dtype=np.float32)

    @cast_backend
    def get_x0(self):
        return self.rng.uniform(low=0.0, high=0.5, size=2)

    @cast_backend
    def get_U(self, nsim, signal=None, **signal_kwargs):
        if signal is not None:
            return super().get_U(nsim=nsim, signal=signal, **signal_kwargs)
        u = step(nsim=nsim, d=1, min=0.0, max=0.03,
                 randsteps=int(np.ceil(nsim/200)), rng=self.rng)
        return u

    @cast_backend
    def equations(self, t, x, u):

        h1 = self.B.core.clip(x[0], 0, 10)  # States (2): level in the tanks
        h2 = self.B.core.clip(x[1], 0, 10)
        q = u[0]  # Inputs (1): valv
        dhdt1 = q/self.F1 - self.k11/self.F1*self.B.core.sqrt(h1) #self.c1 * (1.0 - valve) * pump - self.c2 * self.B.core.sqrt(h1)
        dhdt2 = self.k11/self.F2*self.B.core.sqrt(h1) - self.k22/self.F2*self.B.core.sqrt(h2) #self.c1 * valve * pump + self.c2 * self.B.core.sqrt(h1) - self.c2 * self.B.core.sqrt(h2)

        dhdt = [dhdt1, dhdt2]
        
        return dhdt 