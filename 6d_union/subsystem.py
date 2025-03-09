import heterocl as hcl
import numpy as np
import math

class subsys_1:
    def __init__(self, x=[0,0,0,0], uTMax=1, utMax=1, dMax=0.0, uMode='min', dMode='max'):
        '''
            System dynamics:
                \dot{x} = vx
                \dot{vx} = -uT*sin(theta)
                \dot{theta} = w
                \dot{w} = ut
            uMax: control constraint -uMax <= u <= uMax
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
        T_term = -spat_deriv[1]*math.sin(state[2])
        t_term = spat_deriv[3]
        
        opt_uT = self.uTMax
        opt_ut = self.utMax
        
        if self.uMode == 'min':
            if T_term > 0:
                opt_uT = -opt_uT
            if t_term > 0:
                opt_ut = -opt_ut
        elif self.uMode == 'max':
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
        x_dot = state[1]
        vx_dot = -uOpt[0]*math.sin(state[2])
        theta_dot = state[3]
        w_dot = uOpt[1]
        
        return (x_dot, vx_dot, theta_dot, w_dot)
    
class subsys_2:
    def __init__(self, x=[0,0,0,0], uTMax=1, utMax=1, dMax=0.0, uMode='min', dMode='max'):
        '''
           System dynamics:
                \dot{y} = vy
                \dot{vy} = uT*cos(theta)-g
                \dot{theta} = w
                \dot{w} = ut
            uMax: control constraint -uMax <= u <= uMax
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
        T_term = spat_deriv[1]*math.cos(state[2])
        t_term = spat_deriv[3]
        
        opt_uT = self.uTMax
        opt_ut = self.utMax
        
        if self.uMode == 'min':
            if T_term > 0:
                opt_uT = -opt_uT
            if t_term > 0:
                opt_ut = -opt_ut
        elif self.uMode == 'max':
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
        g = 10
        y_dot = state[1]
        vy_dot = uOpt[0]*math.cos(state[2]) -g
        theta_dot = state[3]
        w_dot = uOpt[1]
        
        return (y_dot, vy_dot, theta_dot, w_dot)
            
    