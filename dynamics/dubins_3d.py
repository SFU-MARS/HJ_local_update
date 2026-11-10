# Numpy dynamics and optimal control of the 3D Dubins car, used by numpy_impl/local_update_3d.py.
import numpy as np
import math

class dubins_car:
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
