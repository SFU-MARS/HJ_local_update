import heterocl as hcl
import numpy as np
import math

class couple_u:
    def __init__(self, x=[0,0], uMax=1, dMax=0.0, uMode='min', dMode='max'):
        '''
            System dynamics: with coupled control
                \dot{x} = ux
                \dot{y} = uy
            uMax: control constraint ux^2 + uy^2 <= uMax^2
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
        # hcl.init(hcl.Float())
        
        opt_ux = hcl.scalar(0, "opt_ux")
        opt_uy = hcl.scalar(0, "opt_uy")
        
        x_term = hcl.scalar(0, "x_term")     
        y_term = hcl.scalar(0, "y_term") 
        
        r_term = hcl.scalar(0, "r_term")
        r_term[0] = hcl.sqrt(spat_deriv[0]*spat_deriv[0]+spat_deriv[1]*spat_deriv[1])+1e-15 # get square root
        
        # print('r_term is', r_term[0])
        
        x_term[0] = spat_deriv[0]/r_term[0]
        y_term[0] = spat_deriv[1]/r_term[0]
        
        opt_ux[0] = x_term[0] * self.uMax
        opt_uy[0] = y_term[0] * self.uMax
        
        with hcl.if_(self.uMode == 'min'):
                opt_ux[0] = -opt_ux[0]
                opt_uy[0] = -opt_uy[0]
                        
        return (opt_ux[0], opt_uy[0])
    
    def opt_ctrl_numpy(self, t, state, spat_deriv):
        '''
        Compute the optimal control at time t
            :param
        '''
        r_term = np.sqrt(spat_deriv[0]*spat_deriv[0]+spat_deriv[1]*spat_deriv[1])+1e-15
        x_term = spat_deriv[0]/r_term
        y_term = spat_deriv[1]/r_term
        
        opt_ux = x_term * self.uMax
        opt_uy = y_term * self.uMax
        
        if self.uMode == 'min':
            opt_ux = -opt_ux
            opt_uy = -opt_uy
            
        return [opt_ux, opt_uy]
    
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
        w_term[0] = spat_deriv[1]
        
        with hcl.if_(w_term[0] >= 0):
            with hcl.if_(self.dMode == 'min'):
                opt_d[0] = -opt_d[0]    
        with hcl.elif_(w_term[0] < 0):
            with hcl.if_(self.dMode == 'max'):
                opt_d[0] = -opt_d[0]
                
        return (opt_d[0], in1[0])
    
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
        y_dot = hcl.scalar(0, "y_dot")
        
        x_dot[0] = uOpt[0]
        y_dot[0] = uOpt[1]
        
        return (x_dot[0], y_dot[0])
    
    def dynamics_numpy(self, t, state, uOpt, dOpt):
        '''
        Compute the system dynamics
            :param
        '''
        x_dot = uOpt[0]
        y_dot = uOpt[1]
        
        return (x_dot, y_dot)
        
    
    

    
                
            