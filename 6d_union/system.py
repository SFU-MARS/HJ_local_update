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
        
        g =10
        
        x_dot = state[2]
        y_dot = state[3]
        vx_dot = -uOpt[0]*math.sin(state[4])
        vy_dot = uOpt[0]*math.cos(state[4]) - g
        theta_dot = state[5]
        w_dot = uOpt[1]
        
        return (x_dot, y_dot, vx_dot, vy_dot, theta_dot, w_dot)
        
    
    

    
                
            