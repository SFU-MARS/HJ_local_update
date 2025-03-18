import heterocl as hcl
import numpy as np
import math

class subsys:
    def __init__(self, x=[0,0,0], uMax=1, dMax=0.0, uMode='min', dMode='max'):
        '''
            System dynamics: with coupled control
                \dot{x} = ux
            uMax: control constraint -uMax <= ux <= uMax
        '''
        self.x = x
        self.uMax = uMax
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode
        
    def opt_ctrl(self, t, state, spat_deriv):
        '''
        Compute the optimal control at time t
            :param
                spat_deriv: the spatial derivative
                state: x
                t: time
            :return
                optimal control
        '''
        opt_ux = hcl.scalar(self.uMax, "opt_ux")
        
        x_term = hcl.scalar(0, "x_term")
        x_term[0] = spat_deriv[0]
        
        opt_ux[0] = self.uMax
        
        with hcl.if_(self.uMode == 'min'):
            with hcl.if_(x_term[0] > 0):
                opt_ux[0] = -opt_ux[0]
        with hcl.elif_(self.uMode == 'max'):
            with hcl.if_(x_term[0] < 0):
                opt_ux[0] = -opt_ux[0]
                
        return (opt_ux[0], )
    
    def opt_ctrl_numpy(self, t, state, spat_deriv):
        '''
        Compute the optimal control at time t
            :param
        '''
        x_term = spat_deriv
        opt_ux = self.uMax
        
        if self.uMode == 'min':
            if x_term > 0:
                opt_ux = -opt_ux
        elif self.uMode == 'max':
            if x_term < 0:
                opt_ux = -opt_ux
                
        return opt_ux
    
    
    def opt_dstb(self, t, state, spat_deriv):
        '''
        Currently no disturbance
        
        Compute the optimal disturbance at time t
            :param
                spat_deriv: tuple of spatial derivative in all dimensions
                state: x, y
                t: time
            :return
                optimal disturbance
        '''
        opt_d = hcl.scalar(0, "opt_d")
        in1 = hcl.scalar(0, "in1")
        
        w_term = hcl.scalar(0, "w_term")
        w_term[0] = spat_deriv[0]
        
        with hcl.if_(w_term[0] >= 0):
            with hcl.if_(self.dMode == 'min'):
                opt_d[0] = -opt_d[0]    
        with hcl.elif_(w_term[0] < 0):
            with hcl.if_(self.dMode == 'max'):
                opt_d[0] = -opt_d[0]
                
        return (opt_d[0], )
    
    def dynamics(self, t, state, uOpt, dOpt):
        '''
        Compute the system dynamics
            :param
                state: x, y
                uOpt: optimal control
                dOpt: optimal disturbance
                t: time
            :return
                x_dot, y_dot
        '''
        x_dot = hcl.scalar(0, "x_dot")
        
        x_dot[0] = uOpt[0]
        
        return (x_dot[0], )
    
    def dynamics_numpy(self, t, state, uOpt, dOpt):
        '''
        Compute the system dynamics
            :param
        '''
        x_dot = uOpt
        
        return x_dot
            
    