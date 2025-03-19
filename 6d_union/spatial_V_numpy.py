import numpy as np

'''
For 4D array
'''
def spa_deriv4X1(i, j, k, l, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 0 not in grid.pDim:
        if i == 0:
            left_deriv = (data[i+1,j,k,l] - data[i,j,k,l])/grid.dx[0]
            right_deriv = (data[i+1,j,k,l] - data[i,j,k,l])/grid.dx[0]
        elif i == data.shape[0]-1:
            left_deriv = (data[i,j,k,l] - data[i-1,j,k,l])/grid.dx[0]
            right_deriv = (data[i,j,k,l] - data[i-1,j,k,l])/grid.dx[0]
        else:
            left_deriv = (data[i,j,k,l] - data[i-1,j,k,l])/grid.dx[0]
            right_deriv = (data[i+1,j,k,l] - data[i,j,k,l])/grid.dx[0]
    else:
        if i == 0:
            left_boundary = data[data.shape[0]-1,j,k,l]
            left_deriv = (data[i,j,k,l] - left_boundary)/grid.dx[0]
            right_deriv = (data[i+1,j,k,l] - data[i,j,k,l])/grid.dx[0]
        elif i == data.shape[0]-1:
            right_boundary = data[0,j,k,l]
            left_deriv = (data[i,j,k,l] - data[i-1,j,k,l])/grid.dx[0]
            right_deriv = (right_boundary - data[i,j,k,l])/grid.dx[0]
        else:
            left_deriv = (data[i,j,k,l] - data[i-1,j,k,l])/grid.dx[0]
            right_deriv = (data[i+1,j,k,l] - data[i,j,k,l])/grid.dx[0]
            
    return left_deriv, right_deriv


def spa_deriv4X2(i, j, k, l, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 1 not in grid.pDim:
        if j == 0:
            left_deriv = (data[i,j+1,k,l] - data[i,j,k,l])/grid.dx[1]
            right_deriv = (data[i,j+1,k,l] - data[i,j,k,l])/grid.dx[1]
        elif j == data.shape[1]-1:
            left_deriv = (data[i,j,k,l] - data[i,j-1,k,l])/grid.dx[1]
            right_deriv = (data[i,j,k,l] - data[i,j-1,k,l])/grid.dx[1]
        else:
            left_deriv = (data[i,j,k,l] - data[i,j-1,k,l])/grid.dx[1]
            right_deriv = (data[i,j+1,k,l] - data[i,j,k,l])/grid.dx[1]
    else:
        if j == 0:
            left_boundary = data[i,data.shape[1]-1,k,l]
            left_deriv = (data[i,j,k,l] - left_boundary)/grid.dx[1]
            right_deriv = (data[i,j+1,k,l] - data[i,j,k,l])/grid.dx[1]
        elif j == data.shape[1]-1:
            right_boundary = data[i,0,k,l]
            left_deriv = (data[i,j,k,l] - data[i,j-1,k,l])/grid.dx[1]
            right_deriv = (right_boundary - data[i,j,k,l])/grid.dx[1]
        else:
            left_deriv = (data[i,j,k,l] - data[i,j-1,k,l])/grid.dx[1]
            right_deriv = (data[i,j+1,k,l] - data[i,j,k,l])/grid.dx[1]
            
    return left_deriv, right_deriv


def spa_deriv4X3(i, j, k, l, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 2 not in grid.pDim:
        if k == 0:
            left_deriv = (data[i,j,k+1,l] - data[i,j,k,l])/grid.dx[2]
            right_deriv = (data[i,j,k+1,l] - data[i,j,k,l])/grid.dx[2]
        elif k == data.shape[2]-1:
            left_deriv = (data[i,j,k,l] - data[i,j,k-1,l])/grid.dx[2]
            right_deriv = (data[i,j,k,l] - data[i,j,k-1,l])/grid.dx[2]
        else:
            left_deriv = (data[i,j,k,l] - data[i,j,k-1,l])/grid.dx[2]
            right_deriv = (data[i,j,k+1,l] - data[i,j,k,l])/grid.dx[2]
    else:
        if k == 0:
            left_boundary = data[i,j,data.shape[2]-1,l]
            left_deriv = (data[i,j,k,l] - left_boundary)/grid.dx[2]
            right_deriv = (data[i,j,k+1,l] - data[i,j,k,l])/grid.dx[2]
        elif k == data.shape[2]-1:
            right_boundary = data[i,j,0,l]
            left_deriv = (data[i,j,k,l] - data[i,j,k-1,l])/grid.dx[2]
            right_deriv = (right_boundary - data[i,j,k,l])/grid.dx[2]
        else:
            left_deriv = (data[i,j,k,l] - data[i,j,k-1,l])/grid.dx[2]
            right_deriv = (data[i,j,k+1,l] - data[i,j,k,l])/grid.dx[2]
            
    return left_deriv, right_deriv


def spa_deriv4X4(i, j, k, l, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 3 not in grid.pDim:
        if l == 0:
            left_deriv = (data[i,j,k,l+1] - data[i,j,k,l])/grid.dx[3]
            right_deriv = (data[i,j,k,l+1] - data[i,j,k,l])/grid.dx[3]
        elif l == data.shape[3]-1:
            left_deriv = (data[i,j,k,l] - data[i,j,k,l-1])/grid.dx[3]
            right_deriv = (data[i,j,k,l] - data[i,j,k,l-1])/grid.dx[3]
        else:
            left_deriv = (data[i,j,k,l] - data[i,j,k,l-1])/grid.dx[3]
            right_deriv = (data[i,j,k,l+1] - data[i,j,k,l])/grid.dx[3]
    else:
        if l == 0:
            left_boundary = data[i,j,k,data.shape[3]-1]
            left_deriv = (data[i,j,k,l] - left_boundary)/grid.dx[3]
            right_deriv = (data[i,j,k,l+1] - data[i,j,k,l])/grid.dx[3]
        elif l == data.shape[3]-1:
            right_boundary = data[i,j,k,0]
            left_deriv = (data[i,j,k,l] - data[i,j,k,l-1])/grid.dx[3]
            right_deriv = (right_boundary - data[i,j,k,l])/grid.dx[3]
        else:
            left_deriv = (data[i,j,k,l] - data[i,j,k,l-1])/grid.dx[3]
            right_deriv = (data[i,j,k,l+1] - data[i,j,k,l])/grid.dx[3]
            
    return left_deriv, right_deriv



'''
For 6D array
'''
def spa_deriv6X1(i, j, k, l, m, n, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 0 not in grid.pDim:
        if i == 0:
            left_deriv = (data[i+1,j,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[0]
            right_deriv = (data[i+1,j,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[0]
        elif i == data.shape[0]-1:
            left_deriv = (data[i,j,k,l,m,n] - data[i-1,j,k,l,m,n])/grid.dx[0]
            right_deriv = (data[i,j,k,l,m,n] - data[i-1,j,k,l,m,n])/grid.dx[0]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i-1,j,k,l,m,n])/grid.dx[0]
            right_deriv = (data[i+1,j,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[0]           
    else:
        if i == 0:
            left_boundary = data[data.shape[0]-1,j,k,l,m,n]
            left_deriv = (data[i,j,k,l,m,n] - left_boundary)/grid.dx[0]
            right_deriv = (data[i+1,j,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[0]
        elif i == data.shape[0]-1:
            right_boundary = data[0,j,k,l,m,n]
            left_deriv = (data[i,j,k,l,m,n] - data[i-1,j,k,l,m,n])/grid.dx[0]
            right_deriv = (right_boundary - data[i,j,k,l,m,n])/grid.dx[0]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i-1,j,k,l,m,n])/grid.dx[0]
            right_deriv = (data[i+1,j,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[0]
            
    return left_deriv, right_deriv



def spa_deriv6X2(i, j, k, l, m, n, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 1 not in grid.pDim:
        if j == 0:
            left_deriv = (data[i,j+1,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[1]
            right_deriv = (data[i,j+1,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[1]
        elif j == data.shape[1]-1:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j-1,k,l,m,n])/grid.dx[1]
            right_deriv = (data[i,j,k,l,m,n] - data[i,j-1,k,l,m,n])/grid.dx[1]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j-1,k,l,m,n])/grid.dx[1]
            right_deriv = (data[i,j+1,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[1]
    else:
        if j == 0:
            left_boundary = data[i,data.shape[1]-1,k,l,m,n]
            left_deriv = (data[i,j,k,l,m,n] - left_boundary)/grid.dx[1]
            right_deriv = (data[i,j+1,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[1]
        elif j == data.shape[1]-1:
            right_boundary = data[i,0,k,l,m,n]
            left_deriv = (data[i,j,k,l,m,n] - data[i,j-1,k,l,m,n])/grid.dx[1]
            right_deriv = (right_boundary - data[i,j,k,l,m,n])/grid.dx[1]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j-1,k,l,m,n])/grid.dx[1]
            right_deriv = (data[i,j+1,k,l,m,n] - data[i,j,k,l,m,n])/grid.dx[1]
            
    return left_deriv, right_deriv



def spa_deriv6X3(i, j, k, l, m, n, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 2 not in grid.pDim:
        if k == 0:
            left_deriv = (data[i,j,k+1,l,m,n] - data[i,j,k,l,m,n])/grid.dx[2]
            right_deriv = (data[i,j,k+1,l,m,n] - data[i,j,k,l,m,n])/grid.dx[2]
        elif k == data.shape[2]-1:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k-1,l,m,n])/grid.dx[2]
            right_deriv = (data[i,j,k,l,m,n] - data[i,j,k-1,l,m,n])/grid.dx[2]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k-1,l,m,n])/grid.dx[2]
            right_deriv = (data[i,j,k+1,l,m,n] - data[i,j,k,l,m,n])/grid.dx[2]    
    else:
        if k == 0:
            left_boundary = data[i,j,data.shape[2]-1,l,m,n]
            left_deriv = (data[i,j,k,l,m,n] - left_boundary)/grid.dx[2]
            right_deriv = (data[i,j,k+1,l,m,n] - data[i,j,k,l,m,n])/grid.dx[2]
        elif k == data.shape[2]-1:
            right_boundary = data[i,j,0,l,m,n]
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k-1,l,m,n])/grid.dx[2]
            right_deriv = (right_boundary - data[i,j,k,l,m,n])/grid.dx[2]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k-1,l,m,n])/grid.dx[2]
            right_deriv = (data[i,j,k+1,l,m,n] - data[i,j,k,l,m,n])/grid.dx[2]
            
    return left_deriv, right_deriv



def spa_deriv6X4(i, j, k, l, m, n, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 3 not in grid.pDim:
        if l == 0:
            left_deriv = (data[i,j,k,l+1,m,n] - data[i,j,k,l,m,n])/grid.dx[3]
            right_deriv = (data[i,j,k,l+1,m,n] - data[i,j,k,l,m,n])/grid.dx[3]
        elif l == data.shape[3]-1:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l-1,m,n])/grid.dx[3]
            right_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l-1,m,n])/grid.dx[3]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l-1,m,n])/grid.dx[3]
            right_deriv = (data[i,j,k,l+1,m,n] - data[i,j,k,l,m,n])/grid.dx[3]
    else:
        if l == 0:
            left_boundary = data[i,j,k,data.shape[3]-1,m,n]
            left_deriv = (data[i,j,k,l,m,n] - left_boundary)/grid.dx[3]
            right_deriv = (data[i,j,k,l+1,m,n] - data[i,j,k,l,m,n])/grid.dx[3]
        elif l == data.shape[3]-1:
            right_boundary = data[i,j,k,0,m,n]
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l-1,m,n])/grid.dx[3]
            right_deriv = (right_boundary - data[i,j,k,l,m,n])/grid.dx[3]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l-1,m,n])/grid.dx[3]
            right_deriv = (data[i,j,k,l+1,m,n] - data[i,j,k,l,m,n])/grid.dx[3]
            
    return left_deriv, right_deriv



def spa_deriv6X5(i, j, k, l, m, n, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 4 not in grid.pDim:
        if m == 0:
            left_deriv = (data[i,j,k,l,m+1,n] - data[i,j,k,l,m,n])/grid.dx[4]
            right_deriv = (data[i,j,k,l,m+1,n] - data[i,j,k,l,m,n])/grid.dx[4]
        elif m == data.shape[4]-1:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m-1,n])/grid.dx[4]
            right_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m-1,n])/grid.dx[4]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m-1,n])/grid.dx[4]
            right_deriv = (data[i,j,k,l,m+1,n] - data[i,j,k,l,m,n])/grid.dx[4]           
    else:
        if m == 0:
            left_boundary = data[i,j,k,l,data.shape[4]-1,n]
            left_deriv = (data[i,j,k,l,m,n] - left_boundary)/grid.dx[4]
            right_deriv = (data[i,j,k,l,m+1,n] - data[i,j,k,l,m,n])/grid.dx[4]
        elif m == data.shape[4]-1:
            right_boundary = data[i,j,k,l,0,n]
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m-1,n])/grid.dx[4]
            right_deriv = (right_boundary - data[i,j,k,l,m,n])/grid.dx[4]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m-1,n])/grid.dx[4]
            right_deriv = (data[i,j,k,l,m+1,n] - data[i,j,k,l,m,n])/grid.dx[4]
            
    return left_deriv, right_deriv



def spa_deriv6X6(i, j, k, l, m, n, data, grid):
    left_deriv = 0
    right_deriv = 0
    
    if 5 not in grid.pDim:
        if n == 0:
            left_deriv = (data[i,j,k,l,m,n+1] - data[i,j,k,l,m,n])/grid.dx[5]
            right_deriv = (data[i,j,k,l,m,n+1] - data[i,j,k,l,m,n])/grid.dx[5]
        elif n == data.shape[5]-1:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m,n-1])/grid.dx[5]
            right_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m,n-1])/grid.dx[5]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m,n-1])/grid.dx[5]
            right_deriv = (data[i,j,k,l,m,n+1] - data[i,j,k,l,m,n])/grid.dx[5]
    else:
        if n == 0:
            left_boundary = data[i,j,k,l,m,data.shape[5]-1]
            left_deriv = (data[i,j,k,l,m,n] - left_boundary)/grid.dx[5]
            right_deriv = (data[i,j,k,l,m,n+1] - data[i,j,k,l,m,n])/grid.dx[5]
        elif n == data.shape[5]-1:
            right_boundary = data[i,j,k,l,m,0]
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m,n-1])/grid.dx[5]
            right_deriv = (right_boundary - data[i,j,k,l,m,n])/grid.dx[5]
        else:
            left_deriv = (data[i,j,k,l,m,n] - data[i,j,k,l,m,n-1])/grid.dx[5]
            right_deriv = (data[i,j,k,l,m,n+1] - data[i,j,k,l,m,n])/grid.dx[5]
            
    return left_deriv, right_deriv