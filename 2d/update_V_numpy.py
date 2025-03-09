import numpy as np

'''
For 1D array
'''
def spa_deriv(i, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 0 not in grid.pDim:
        if i == 0:
            left_deriv = (data[i+1] - data[i])/grid.dx[0]
            right_deriv = (data[i+1] - data[i])/grid.dx[0]
        elif i == data.shape[0]-1:
            left_deriv = (data[i] - data[i-1])/grid.dx[0]
            right_deriv = (data[i] - data[i-1])/grid.dx[0]
        else:
            left_deriv = (data[i] - data[i-1])/grid.dx[0]
            right_deriv = (data[i+1] - data[i])/grid.dx[0]
            
    else:
        if i == 0:
            left_boundary = data[data.shape[0]-1]
            left_deriv = (data[i] - left_boundary)/grid.dx[0]
            right_deriv = (data[i+1] - data[i])/grid.dx[0]
        elif i == data.shape[0]-1:
            right_boundary = data[0]
            left_deriv = (data[i] - data[i-1])/grid.dx[0]
            right_deriv = (right_boundary - data[i])/grid.dx[0]
        else:
            left_deriv = (data[i] - data[i-1])/grid.dx[0]
            right_deriv = (data[i+1] - data[i])/grid.dx[0]
            
    return left_deriv, right_deriv




'''
For 2D array
'''
def spa_derivX(i, j, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 0 not in grid.pDim:
        if i == 0:
            left_deriv = (data[i+1,j] - data[i,j])/grid.dx[0]
            right_deriv = (data[i+1,j] - data[i,j])/grid.dx[0]
        elif i == data.shape[0]-1:
            left_deriv = (data[i,j] - data[i-1,j])/grid.dx[0]
            right_deriv = (data[i,j] - data[i-1,j])/grid.dx[0]
        else:
            left_deriv = (data[i,j] - data[i-1,j])/grid.dx[0]
            right_deriv = (data[i+1,j] - data[i,j])/grid.dx[0]
            
    else:
        if i == 0:
            left_boundary = data[data.shape[0]-1,j]
            left_deriv = (data[i,j] - left_boundary)/grid.dx[0]
            right_deriv = (data[i+1,j] - data[i,j])/grid.dx[0]
        elif i == data.shape[0]-1:
            right_boundary = data[0,j]
            left_deriv = (data[i,j] - data[i-1,j])/grid.dx[0]
            right_deriv = (right_boundary - data[i,j])/grid.dx[0]
        else:
            left_deriv = (data[i,j] - data[i-1,j])/grid.dx[0]
            right_deriv = (data[i+1,j] - data[i,j])/grid.dx[0]
            
    return left_deriv, right_deriv

def spa_derivY(i, j, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 1 not in grid.pDim:
        if j == 0:
            left_deriv = (data[i,j+1] - data[i,j])/grid.dx[1]
            right_deriv = (data[i,j+1] - data[i,j])/grid.dx[1]
        elif j == data.shape[1]-1:
            left_deriv = (data[i,j] - data[i,j-1])/grid.dx[1]
            right_deriv = (data[i,j] - data[i,j-1])/grid.dx[1]
        else:
            left_deriv = (data[i,j] - data[i,j-1])/grid.dx[1]
            right_deriv = (data[i,j+1] - data[i,j])/grid.dx[1]
            
    else:
        if j == 0:
            left_boundary = data[i,data.shape[1]-1]
            left_deriv = (data[i,j] - left_boundary)/grid.dx[1]
            right_deriv = (data[i,j+1] - data[i,j])/grid.dx[1]
        elif j == data.shape[1]-1:
            right_boundary = data[i,0]
            left_deriv = (data[i,j] - data[i,j-1])/grid.dx[1]
            right_deriv = (right_boundary - data[i,j])/grid.dx[1]
        else:
            left_deriv = (data[i,j] - data[i,j-1])/grid.dx[1]
            right_deriv = (data[i,j+1] - data[i,j])/grid.dx[1]
            
    return left_deriv, right_deriv