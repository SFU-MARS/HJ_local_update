# 3D subsystems of the 4D Dubins car (x, y, v, theta) with speed limits v in [0, 2], used by the 4d decomposition:
#   subsys_1: (x, v, theta),  subsys_2: (y, v, theta)
# Each has HeteroCL methods (for HJSolver) and numpy methods.
import heterocl as hcl
import math


class subsys_1:
    def __init__(self, x=[0, 0, 0], aMax=1, wMax=1, vMin=0.0, vMax=2.0, dMax=0.0, uMode='min', dMode='max'):
        r'''
            System dynamics:
                \dot{x} = v*cos(theta)
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

    def opt_ctrl(self, t, state, spat_deriv):
        opt_a = hcl.scalar(self.aMax, "opt_a")
        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        with hcl.if_(spat_deriv[1] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_a[0] = -opt_a
        with hcl.elif_(spat_deriv[1] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_a[0] = -opt_a
        with hcl.if_(spat_deriv[2] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[2] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
        return (opt_a[0], opt_w[0], in3[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        v_dot = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = state[1] * hcl.cos(state[2])
        v_dot[0] = uOpt[0]
        # Speed limits: no braking below vMin, no acceleration above vMax
        with hcl.if_(hcl.or_(hcl.and_(state[1] <= self.vMin, v_dot[0] < 0),
                             hcl.and_(state[1] >= self.vMax, v_dot[0] > 0))):
            v_dot[0] = 0
        theta_dot[0] = uOpt[1]

        return (x_dot[0], v_dot[0], theta_dot[0])

    def opt_ctrl_numpy(self, t, state, spat_deriv):
        a_term = spat_deriv[1]
        w_term = spat_deriv[2]
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
        x_dot = state[1] * math.cos(state[2])
        v_dot = uOpt[0]
        if (state[1] <= self.vMin and v_dot < 0) or (state[1] >= self.vMax and v_dot > 0):
            v_dot = 0  # speed limits: no braking below vMin, no acceleration above vMax
        theta_dot = uOpt[1]

        return (x_dot, v_dot, theta_dot)


class subsys_2:
    def __init__(self, x=[0, 0, 0], aMax=1, wMax=1, vMin=0.0, vMax=2.0, dMax=0.0, uMode='min', dMode='max'):
        r'''
            System dynamics:
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

    def opt_ctrl(self, t, state, spat_deriv):
        opt_a = hcl.scalar(self.aMax, "opt_a")
        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        with hcl.if_(spat_deriv[1] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_a[0] = -opt_a
        with hcl.elif_(spat_deriv[1] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_a[0] = -opt_a
        with hcl.if_(spat_deriv[2] > 0):
            with hcl.if_(self.uMode == "min"):
                opt_w[0] = -opt_w
        with hcl.elif_(spat_deriv[2] < 0):
            with hcl.if_(self.uMode == "max"):
                opt_w[0] = -opt_w
        return (opt_a[0], opt_w[0], in3[0])

    def opt_dstb(self, t, state, spat_deriv):
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        y_dot = hcl.scalar(0, "y_dot")
        v_dot = hcl.scalar(0, "v_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        y_dot[0] = state[1] * hcl.sin(state[2])
        v_dot[0] = uOpt[0]
        # Speed limits: no braking below vMin, no acceleration above vMax
        with hcl.if_(hcl.or_(hcl.and_(state[1] <= self.vMin, v_dot[0] < 0),
                             hcl.and_(state[1] >= self.vMax, v_dot[0] > 0))):
            v_dot[0] = 0
        theta_dot[0] = uOpt[1]

        return (y_dot[0], v_dot[0], theta_dot[0])

    def opt_ctrl_numpy(self, t, state, spat_deriv):
        a_term = spat_deriv[1]
        w_term = spat_deriv[2]
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
        y_dot = state[1] * math.sin(state[2])
        v_dot = uOpt[0]
        if (state[1] <= self.vMin and v_dot < 0) or (state[1] >= self.vMax and v_dot > 0):
            v_dot = 0  # speed limits: no braking below vMin, no acceleration above vMax
        theta_dot = uOpt[1]

        return (y_dot, v_dot, theta_dot)
