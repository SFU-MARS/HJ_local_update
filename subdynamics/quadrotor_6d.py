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

    def opt_ctrl(self, t, state, spat_deriv):
        '''
        HeteroCL version of opt_ctrl_numpy (optimized_dp's 4D HJSolver unpacks 4 controls)
        '''
        T_term = hcl.scalar(0, "T_term")
        T_term[0] = -spat_deriv[1]*hcl.sin(state[2])
        opt_uT = hcl.scalar(self.uTMax, "opt_uT")
        opt_ut = hcl.scalar(self.utMax, "opt_ut")
        if self.uMode == 'min':
            with hcl.if_(T_term[0] > 0):
                opt_uT[0] = -opt_uT[0]
            with hcl.if_(spat_deriv[3] > 0):
                opt_ut[0] = -opt_ut[0]
        else:
            with hcl.if_(T_term[0] < 0):
                opt_uT[0] = -opt_uT[0]
            with hcl.if_(spat_deriv[3] < 0):
                opt_ut[0] = -opt_ut[0]
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        return (opt_uT[0], opt_ut[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        '''
        No disturbance (optimized_dp's 4D HJSolver unpacks 4 disturbances)
        '''
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        '''
        HeteroCL version of dynamics_numpy (g = 0)
        '''
        v_dot = hcl.scalar(0, "v_dot")
        v_dot[0] = -uOpt[0]*hcl.sin(state[2])
        return (state[1], v_dot[0], state[3], uOpt[1])


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
        g = 0
        
        y_dot = state[1]
        vy_dot = uOpt[0]*math.cos(state[2]) -g
        theta_dot = state[3]
        w_dot = uOpt[1]
        
        return (y_dot, vy_dot, theta_dot, w_dot)

    def opt_ctrl(self, t, state, spat_deriv):
        '''
        HeteroCL version of opt_ctrl_numpy (optimized_dp's 4D HJSolver unpacks 4 controls)
        '''
        T_term = hcl.scalar(0, "T_term")
        T_term[0] = spat_deriv[1]*hcl.cos(state[2])
        opt_uT = hcl.scalar(self.uTMax, "opt_uT")
        opt_ut = hcl.scalar(self.utMax, "opt_ut")
        if self.uMode == 'min':
            with hcl.if_(T_term[0] > 0):
                opt_uT[0] = -opt_uT[0]
            with hcl.if_(spat_deriv[3] > 0):
                opt_ut[0] = -opt_ut[0]
        else:
            with hcl.if_(T_term[0] < 0):
                opt_uT[0] = -opt_uT[0]
            with hcl.if_(spat_deriv[3] < 0):
                opt_ut[0] = -opt_ut[0]
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        return (opt_uT[0], opt_ut[0], in3[0], in4[0])

    def opt_dstb(self, t, state, spat_deriv):
        '''
        No disturbance (optimized_dp's 4D HJSolver unpacks 4 disturbances)
        '''
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        d3 = hcl.scalar(0, "d3")
        d4 = hcl.scalar(0, "d4")
        return (d1[0], d2[0], d3[0], d4[0])

    def dynamics(self, t, state, uOpt, dOpt):
        '''
        HeteroCL version of dynamics_numpy (g = 0)
        '''
        v_dot = hcl.scalar(0, "v_dot")
        v_dot[0] = uOpt[0]*hcl.cos(state[2])
        return (state[1], v_dot[0], state[3], uOpt[1])
