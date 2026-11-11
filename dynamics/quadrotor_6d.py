# Dynamics and optimal control of the 6D quadrotor, numpy methods (numpy_impl/local_update_6d.py,
# compute/direct_6d.py) and HeteroCL methods (hcl_impl/local_update_6d.py, same arithmetic).
import heterocl as hcl
import numpy as np
import math

class quadrotor:
    def __init__(self, x=[0,0,0,0,0,0], uTMax=1, utMax=1, dMax=0.0, uMode='min', dMode='max'):
        '''
            System dynamics:
                \dot{x} = vx
                \dot{y} = vy
                \dot{vx} = -uT*sin(theta)
                \dot{vy} = uT*cos(theta)-g
                \dot{theta} = w
                \dot{w} = ut
        '''
        self.x = x
        self.uTMax = uTMax
        self.utMax = utMax
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode
        
    
    def opt_ctrl_numpy(self, t, state, spat_deriv):
        '''
        Compute the optimal control at time t
            :param
        '''
        
        T_term = -spat_deriv[2]*math.sin(state[4]) + spat_deriv[3]*math.cos(state[4])
        t_term = spat_deriv[5]
        
        opt_uT = self.uTMax
        opt_ut = self.utMax
        
        if self.uMode == 'min':
            if T_term > 0:
                opt_uT = -opt_uT
            if t_term > 0:
                opt_ut = -opt_ut
        else:
            if T_term < 0:
                opt_uT = -opt_uT
            if t_term < 0:
                opt_ut = -opt_ut

        return (opt_uT, opt_ut)

    
    def dynamics_numpy(self, t, state, uOpt, dOpt):
        '''
        Compute the system dynamics
            :param
        '''
        
        g = 0
        
        x_dot = state[2]
        y_dot = state[3]
        vx_dot = -uOpt[0]*math.sin(state[4])
        vy_dot = uOpt[0]*math.cos(state[4]) - g
        theta_dot = state[5]
        w_dot = uOpt[1]
        
        return (x_dot, y_dot, vx_dot, vy_dot, theta_dot, w_dot)

    def opt_ctrl(self, t, state, spat_deriv):
        '''
        HeteroCL version of opt_ctrl_numpy
        '''
        T_term = hcl.scalar(0, "T_term")
        T_term[0] = -spat_deriv[2]*hcl.sin(state[4]) + spat_deriv[3]*hcl.cos(state[4])
        opt_uT = hcl.scalar(self.uTMax, "opt_uT")
        opt_ut = hcl.scalar(self.utMax, "opt_ut")
        if self.uMode == 'min':
            with hcl.if_(T_term[0] > 0):
                opt_uT[0] = -opt_uT[0]
            with hcl.if_(spat_deriv[5] > 0):
                opt_ut[0] = -opt_ut[0]
        else:
            with hcl.if_(T_term[0] < 0):
                opt_uT[0] = -opt_uT[0]
            with hcl.if_(spat_deriv[5] < 0):
                opt_ut[0] = -opt_ut[0]
        return (opt_uT[0], opt_ut[0])

    def dynamics(self, t, state, uOpt, dOpt):
        '''
        HeteroCL version of dynamics_numpy (g = 0)
        '''
        vx_dot = hcl.scalar(0, "vx_dot")
        vy_dot = hcl.scalar(0, "vy_dot")
        vx_dot[0] = -uOpt[0]*hcl.sin(state[4])
        vy_dot[0] = uOpt[0]*hcl.cos(state[4])
        return (state[2], state[3], vx_dot[0], vy_dot[0], state[5], uOpt[1])
