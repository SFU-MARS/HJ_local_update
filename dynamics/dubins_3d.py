# Dynamics and optimal control of the 3D Dubins car, numpy methods (numpy_impl/local_update_3d.py) and
# HeteroCL methods (hcl_impl/local_update_3d.py, same as DubinsCar in optimized_dp).
import heterocl as hcl
import numpy as np
import math

from odp.computeGraphs.CustomGraphFunctions import my_abs, my_max

class dubins_car:
    # The dissipation coefficients do not depend on the derivative bounds (|w| = wMax either way)
    DISSIPATION_USES_DERIV_BOUNDS = False

    def __init__(self, x=[0,0,0], v=1, wMax=1, dMax=0.0, uMode='min', dMode='max'):
        '''
            System dynamics:
                \dot{x} = v*cos(theta)
                \dot{y} = v*sin(theta)
                \dot{theta} = w
            v: constant speed
            wMax: control constraint |w| <= wMax
        '''
        self.x = x
        self.v = v
        self.wMax = wMax
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode


    def opt_ctrl_numpy(self, t, state, spat_deriv):
        '''
        Compute the optimal control at time t
            :param
        '''

        w_term = spat_deriv[2]

        opt_w = self.wMax

        if self.uMode == 'min':
            if w_term > 0:
                opt_w = -opt_w
        else:
            if w_term < 0:
                opt_w = -opt_w

        return (opt_w,)


    def dynamics_numpy(self, t, state, uOpt, dOpt):
        '''
        Compute the system dynamics
            :param
        '''

        x_dot = self.v*math.cos(state[2])
        y_dot = self.v*math.sin(state[2])
        theta_dot = uOpt[0]

        return (x_dot, y_dot, theta_dot)

    def dissipation_coeffs_numpy(self, t, state, deriv_min=None, deriv_max=None):
        '''
        Lax-Friedrichs dissipation coefficients: the max |rate of change| of each state over the
        control set |w| <= wMax, same as optimized_dp
        '''
        rates_1 = self.dynamics_numpy(t, state, (self.wMax,), 0)
        rates_2 = self.dynamics_numpy(t, state, (-self.wMax,), 0)
        return np.maximum(np.abs(rates_1), np.abs(rates_2))

    def opt_ctrl(self, t, state, spat_deriv):
        '''
        HeteroCL version of opt_ctrl_numpy (same as DubinsCar.opt_ctrl in optimized_dp)
        '''
        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        with hcl.if_(spat_deriv[2] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[2] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
        return (opt_w[0], in3[0], in4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        '''
        HeteroCL version of dynamics_numpy (same as DubinsCar.dynamics in optimized_dp)
        '''
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = self.v*hcl.cos(state[2])
        y_dot[0] = self.v*hcl.sin(state[2])
        theta_dot[0] = uOpt[0]

        return (x_dot[0], y_dot[0], theta_dot[0])

    def dissipation_coeffs(self, t, state, deriv_min=None, deriv_max=None):
        '''
        HeteroCL version of dissipation_coeffs_numpy
        '''
        r1_x, r1_y, r1_T = self.dynamics(t, state, (self.wMax, 0, 0), (0, 0, 0))
        r2_x, r2_y, r2_T = self.dynamics(t, state, (-self.wMax, 0, 0), (0, 0, 0))
        alpha_x = hcl.scalar(0, "alpha_x")
        alpha_y = hcl.scalar(0, "alpha_y")
        alpha_T = hcl.scalar(0, "alpha_T")
        alpha_x[0] = my_max(my_abs(r1_x), my_abs(r2_x))
        alpha_y[0] = my_max(my_abs(r1_y), my_abs(r2_y))
        alpha_T[0] = my_max(my_abs(r1_T), my_abs(r2_T))
        return (alpha_x[0], alpha_y[0], alpha_T[0])
