# Dynamics and optimal control of the 4D Dubins car, numpy methods (numpy_impl/local_update_4d.py) and HeteroCL
# methods (compute/direct_4d.py, hcl_impl/local_update_4d.py): DubinsCar4D in optimized_dp without disturbance,
# except for the speed limits: v stays in [vMin, vMax] (no reversing, top speed).
import heterocl as hcl
import numpy as np
import math


class dubins_car_4d:
    # At v = vMin and v = vMax the rate of v depends on the sign of the optimal acceleration, so the dissipation coefficients
    # depend on the global derivative bounds (as in HJSolver)
    DISSIPATION_USES_DERIV_BOUNDS = True
    # Only the bounds of dV/dv matter: x and y have no control, and |w| = wMax whatever the sign of dV/dtheta,
    # so the local update computes the bounds of dimension 2 only (the others are set to 0)
    DERIV_BOUNDS_DIMS = (2,)

    def __init__(self, x=[0, 0, 0, 0], aMax=1, wMax=1, vMin=0.0, vMax=2.0, dMax=0.0, uMode='min', dMode='max'):
        r'''
            State (x, y, v, theta), system dynamics:
                \dot{x} = v*cos(theta)
                \dot{y} = v*sin(theta)
                \dot{v} = a, but 0 if a < 0 at v <= vMin or a > 0 at v >= vMax (speed limits)
                \dot{theta} = w
            aMax, wMax: control constraints |a| <= aMax, |w| <= wMax
            vMin, vMax: lowest speed (0: no reversing) and top speed
        '''
        self.x = x
        self.aMax = aMax
        self.wMax = wMax
        self.vMin = vMin
        self.vMax = vMax
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl_numpy(self, t, state, spat_deriv):
        '''
        Compute the optimal control at time t: bang-bang on the signs of dV/dv (a) and dV/dtheta (w)
        '''
        a_term = spat_deriv[2]
        w_term = spat_deriv[3]

        opt_a = self.aMax
        opt_w = self.wMax

        if self.uMode == 'min':
            if a_term > 0:
                opt_a = -opt_a
            if w_term > 0:
                opt_w = -opt_w
        else:
            if a_term < 0:
                opt_a = -opt_a
            if w_term < 0:
                opt_w = -opt_w

        return (opt_a, opt_w)

    def dynamics_numpy(self, t, state, uOpt, dOpt):
        '''
        Compute the system dynamics (in 32-bit floats like HJSolver: cos/sin of theta, then times v)
        '''
        f = np.float32
        x_dot = f(state[2]) * f(math.cos(state[3]))
        y_dot = f(state[2]) * f(math.sin(state[3]))
        v_dot = uOpt[0]
        if (state[2] <= self.vMin and v_dot < 0) or (state[2] >= self.vMax and v_dot > 0):
            v_dot = 0  # speed limits: no braking below vMin, no acceleration above vMax
        theta_dot = uOpt[1]

        return (x_dot, y_dot, v_dot, theta_dot)

    def dissipation_coeffs_numpy(self, t, state, deriv_min=None, deriv_max=None):
        '''
        Lax-Friedrichs dissipation coefficients as in optimized_dp's HJSolver: the largest |rate of change|
        of each state over the optimal controls at the global min and max spatial derivatives
        '''
        rates_L = self.dynamics_numpy(t, state, self.opt_ctrl_numpy(t, state, deriv_min), 0)
        rates_U = self.dynamics_numpy(t, state, self.opt_ctrl_numpy(t, state, deriv_max), 0)
        return [max(abs(a), abs(b)) for a, b in zip(rates_L, rates_U)]

    def opt_ctrl(self, t, state, spat_deriv):
        '''
        HeteroCL version of opt_ctrl_numpy (same as DubinsCar4D.opt_ctrl in optimized_dp)
        '''
        opt_a = hcl.scalar(self.aMax, "opt_a")
        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        if self.uMode == 'min':
            with hcl.if_(spat_deriv[2] > 0):
                opt_a[0] = -opt_a[0]
            with hcl.if_(spat_deriv[3] > 0):
                opt_w[0] = -opt_w[0]
        else:
            with hcl.if_(spat_deriv[2] < 0):
                opt_a[0] = -opt_a[0]
            with hcl.if_(spat_deriv[3] < 0):
                opt_w[0] = -opt_w[0]
        return (opt_a[0], opt_w[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        '''
        No disturbance (optimized_dp's 4D HJSolver unpacks 4 disturbances)
        '''
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        '''
        HeteroCL version of dynamics_numpy (same as DubinsCar4D.dynamics in optimized_dp)
        '''
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        v_dot = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = state[2] * hcl.cos(state[3])
        y_dot[0] = state[2] * hcl.sin(state[3])
        v_dot[0] = uOpt[0]
        # Speed limits: no braking below vMin, no acceleration above vMax
        with hcl.if_(hcl.or_(hcl.and_(state[2] <= self.vMin, v_dot[0] < 0),
                             hcl.and_(state[2] >= self.vMax, v_dot[0] > 0))):
            v_dot[0] = 0
        theta_dot[0] = uOpt[1]

        return (x_dot[0], y_dot[0], v_dot[0], theta_dot[0])

    def dissipation_coeffs(self, t, state, deriv_min=None, deriv_max=None):
        '''
        HeteroCL version of dissipation_coeffs_numpy
        '''
        rates_L = self.dynamics(t, state, self.opt_ctrl(t, state, deriv_min), (0, 0, 0, 0))
        rates_U = self.dynamics(t, state, self.opt_ctrl(t, state, deriv_max), (0, 0, 0, 0))
        alpha = []
        for d, (a, b) in enumerate(zip(rates_L, rates_U)):
            abs_a = hcl.select(a > 0, a, -a)
            abs_b = hcl.select(b > 0, b, -b)
            alpha_d = hcl.scalar(0, f"alpha{d}")
            alpha_d[0] = hcl.select(abs_a > abs_b, abs_a, abs_b)
            alpha.append(alpha_d[0])
        return tuple(alpha)
