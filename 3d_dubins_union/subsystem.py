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
