from direct_hcl import direct_comp
from decomposition_hcl import decomposition

from set_2plot import plot_overlay_set
from odp.Plots import PlotOptions
import numpy as np
from odp.Plots import plot_isosurface, plot_valuefunction

import plotly.express as px

from odp.Grid import Grid
from odp.Shapes import *

from system import couple_u

# The function of computing first order spatial derivative
from update_V_hcl import update_V

import heterocl as hcl
import time

num = 101

grid, result_true = direct_comp(num)
result_decomp = decomposition(num)

true_final = result_true[:, :, 0]
decomp_final = result_decomp[:, :, 0]
result_diff = true_final - decomp_final

fig = px.imshow(result_diff)
# fig.show()

indices = np.argwhere(abs(result_diff) > 0)
print(indices[1,1])

# change the input (result_diff) to find different dictionary to get updated
# dict_from_array = dict(map(lambda x: (tuple(x[1]), result_diff[tuple(x[1])]), enumerate(indices)))

# look=back length and time step of computation
lookback_length = 0.2
t_step = 0.02
small_number = 1e-5
tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

sys = couple_u(x=[0, 0], uMax=1, dMax=0.0, uMode='min', dMode='min')

# Create Grid
grid_min = np.array([-4.0, -4.0])
grid_max = np.array([4.0, 4.0])
dims = grid_min.shape[0]
N = np.array([num, num])
g = Grid(grid_min, grid_max, dims, N)

## Initialize value function
data = ShapeRectangle(g, [-1.0, -1.0], [1.0, 1.0])

# change the input (result_diff) to find different dictionary to get updated
dict_to_update = dict(map(lambda x: (tuple(x[1]), data[tuple(x[1])]), enumerate(indices)))
# print(dict_to_update)

po = PlotOptions(do_plot=True, plot_type="value", plotDims=[0, 1],
                        colorscale="Bluered", save_fig=False, filename="direct_2Int_leaking", interactive_html=True)

'''
Below is how to locally update the value function with respect to the indices
'''
hcl.init()
hcl.config.init_dtype = hcl.Float(32)

print("Initializing\n")

init_value = data

# Array of each state value
list_x1 = np.reshape(g.vs[0], g.pts_each_dim[0])
list_x2 = np.reshape(g.vs[1], g.pts_each_dim[1])
# Convert state array to hcl array type
list_x1 = hcl.asarray(list_x1)
list_x2 = hcl.asarray(list_x2)

# Tensors input to our computation graph
V_0 = hcl.asarray(init_value)
V_1 = hcl.asarray(np.zeros(tuple(g.pts_each_dim)))
ind = hcl.asarray(indices)
print(ind.asnumpy()[0,1], 'ind.asnumpy pass')

# Get exexutable
solve_local_pde = update_V(sys, g, ind)

## No option for saving all time steps for now

execution_time = 0
iter = 0
tNow = tau[0]
print("Started running\n")

for i in range (1, len(tau)):

    t_local = hcl.asarray(np.array((tNow, tau[i])))
    print(t_local, 't_local pass')

    while tNow < tau[i] - 1e-4:
        prev_arr = V_0.asnumpy()
        # Start timing
        iter += 1
        start = time.time()

        solve_local_pde(V_1, V_0, list_x1, list_x2, ind, t_local)

        tNow = t_local.asnumpy()[1]

        execution_time += time.time() - start

        print("Iteration: ", i)

        print(t_local)
        print("Computational time to integrate (s): {:.5f}".format(time.time() - start))

print("Total kernel time (s): {:.5f}".format(execution_time))
print("Finished solving\n")

plot_valuefunction(grid, V_1.asnumpy(), po)






# for key in dict_to_update.keys():
#     print(key, "->", dict_to_update[key])




            


