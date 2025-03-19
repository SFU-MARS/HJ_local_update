import numpy as np

def ground_truth_L(data_true, data_approx):
    '''
    Get the ground truth of the leaking corners
    '''
    result_diff = data_approx - data_true
    indice = np.argwhere(abs(result_diff) > 1e-6)
    return indice
    
def new_theory_L(data_approx, data_init, data_upper, data_lower):
    '''
    Test the new theory of finding the leaking corners
    '''
    # delta_1 = np.max(abs(data_upper[0] - data_upper[-1]))
    # delta_2 = np.max(abs(data_lower[0] - data_lower[-1]))
    # print('The delta 1 value is', delta_1)
    # print('The delta 2 value is', delta_2)
    # delta = max(delta_1,delta_2)
    
    delta = np.max(abs(data_init - data_approx), axis = (1,2))
    
    t_len = data_approx.shape[0]
    
    print('The delta value is', delta)
    indice = []
    for i in range(t_len):
        indice.append(np.argwhere(abs(data_upper[i] - data_lower[i]) <= delta[i]))
    # indice = np.array(indice)
    # print(indice.shape)
    return indice