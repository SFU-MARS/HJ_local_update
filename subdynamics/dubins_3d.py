# 2D subsystems of the 3D Dubins car, used by compute/decomposition_3d.py:
#   subsys_1: (x, theta),  subsys_2: (y, theta)
# Each has HeteroCL methods (for HJSolver) and numpy methods.
import heterocl as hcl
import math


class subsys_1:
    def __init__(self, x=[0, 0], v=1, wMax=1, dMax=0.0, uMode='min', dMode='max'):
        r'''
            System dynamics:
                \dot{x} = v*cos(theta)
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

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in2 = hcl.scalar(0, "in2")
        with hcl.if_(spat_deriv[1] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[1] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
        return (opt_w[0], in2[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        return (d1[0], d2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = self.v * hcl.cos(state[1])
        theta_dot[0] = uOpt[0]

        return (x_dot[0], theta_dot[0])

    def opt_ctrl_numpy(self, t, state, spat_deriv):
        w_term = spat_deriv[1]
        opt_w = self.wMax

        if self.uMode == 'min':
            if w_term > 0:
                opt_w = -opt_w
        else:
            if w_term < 0:
                opt_w = -opt_w

        return (opt_w,)

    def dynamics_numpy(self, t, state, uOpt, dOpt):
        x_dot = self.v * math.cos(state[1])
        theta_dot = uOpt[0]

        return (x_dot, theta_dot)


class subsys_2:
    def __init__(self, x=[0, 0], v=1, wMax=1, dMax=0.0, uMode='min', dMode='max'):
        r'''
            System dynamics:
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

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in2 = hcl.scalar(0, "in2")
        with hcl.if_(spat_deriv[1] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[1] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
        return (opt_w[0], in2[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        return (d1[0], d2[0])

    def dynamics(self, t, state, uOpt, dOpt):
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        y_dot[0] = self.v * hcl.sin(state[1])
        theta_dot[0] = uOpt[0]

        return (y_dot[0], theta_dot[0])

    def opt_ctrl_numpy(self, t, state, spat_deriv):
        w_term = spat_deriv[1]
        opt_w = self.wMax

        if self.uMode == 'min':
            if w_term > 0:
                opt_w = -opt_w
        else:
            if w_term < 0:
                opt_w = -opt_w

        return (opt_w,)

    def dynamics_numpy(self, t, state, uOpt, dOpt):
        y_dot = self.v * math.sin(state[1])
        theta_dot = uOpt[0]

        return (y_dot, theta_dot)
