import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from odp.Grid import Grid
from odp.Plots import pre_plot
from odp.Plots import PlotOptions
import numpy as np
import plotly.express as px


def plot_overlay_set(g, data1, data2, data3, plot_option):
    dims_plot = plot_option.dims_plot
    grid1, my_V1 = pre_plot(plot_option, g, data1)
    grid2, my_V2 = pre_plot(plot_option, g, data2)
    grid2, my_V3 = pre_plot(plot_option, g, data3)

    # my_V = np.zeros(my_V1.shape)
    # np.where(my_V1 != 0, 0, 1)
    # np.where(my_V2 != 0, 0, 2)

    # my_V = np.maximum(my_V1, my_V2)
    
    if len(dims_plot) == 2:
        # plot 2 2D sets
        # dim1, dim2 = dims_plot[0], dims_plot[1]
        complex_x = complex(0, grid1.pts_each_dim[dims_plot[0]])
        complex_y = complex(0, grid1.pts_each_dim[dims_plot[1]])
        mg_X, mg_Y = np.mgrid[grid1.min[0]:grid1.max[0]: complex_x, grid1.min[1]:grid1.max[1]: complex_y]
        
        # N = my_V1.shape[2]
        
        
        trace1 = go.Contour(
                    x=mg_X.flatten(),
                    y=mg_Y.flatten(),
                    z=my_V1.flatten(),
                    zmin=0.0,
                    ncontours=1,
                    contours_coloring='none',  # former: lines
                    name="Direct Computation",  # zero level
                    line_width=1.5,
                    line_color='magenta',
                    zmax=0.0,
                )

       
        trace2 = go.Contour(
                    x=mg_X.flatten(),
                    y=mg_Y.flatten(),
                    z=my_V2.flatten(),
                    zmin=0.0,
                    ncontours=1,
                    contours_coloring='none',  # former: lines
                    name="System Decomposition",  # zero level
                    line_width=1.5,
                    line_color='blue',
                    zmax=0.0,
                )
        
        trace3 = go.Contour(
                    x=mg_X.flatten(),
                    y=mg_Y.flatten(),
                    z=my_V3.flatten(),
                    zmin=0.0,
                    ncontours=1,
                    contours_coloring='none',  # former: lines
                    name="Local",  # zero level
                    line_width=1.5,
                    line_color='green',
                    zmax=0.0,
                )
        
        fig = go.Figure()
        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.add_trace(trace3)
        fig.update_layout(
            title='2D Value Function',
            scene=dict( xaxis={"nticks": 20},
                        zaxis={"nticks": 20},
                        camera_eye={"x": 0, "y": -1, "z": 0.5},
                        aspectratio={"x": 1, "y": 1, "z": 0.6}
                        ))
        fig.show()
        
        
    if len(dims_plot) == 3:
        # plot 2 3D sets
        dim1, dim2, dim3 = dims_plot[0], dims_plot[1], dims_plot[2]
        complex_x = complex(0, grid1.pts_each_dim[dim1])
        complex_y = complex(0, grid1.pts_each_dim[dim2])
        complex_z = complex(0, grid1.pts_each_dim[dim3])
        X, Y, Z = np.mgrid[grid1.min[dim1]:grid1.max[dim1]:complex_x,
                        grid1.min[dim2]:grid1.max[dim2]:complex_y,
                        grid1.min[dim3]:grid1.max[dim3]:complex_z]
        
        trace1 = go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=my_V1.flatten(),
            isomin=plot_option.min_isosurface,
            isomax=plot_option.max_isosurface,
            surface_count=plot_option.surface_count,
            colorscale="earth",
            opacity=1,
            contour=plot_option.contour,
            flatshading=plot_option.flatshading,
            lighting=plot_option.lighting,
            lightposition=plot_option.lightposition,
            reversescale=plot_option.reversescale,
            showlegend=plot_option.showlegend,
            showscale=plot_option.showscale,
        )

        trace2 = go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=my_V2.flatten(),
            isomin=plot_option.min_isosurface,
            isomax=plot_option.max_isosurface,
            surface_count=plot_option.surface_count,
            colorscale="sunsetdark",
            opacity=0.3,
            contour=plot_option.contour,
            flatshading=plot_option.flatshading,
            lighting=plot_option.lighting,
            lightposition=plot_option.lightposition,
            reversescale=plot_option.reversescale,
            showlegend=plot_option.showlegend,
            showscale=plot_option.showscale,
        )

        # fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig = go.Figure()
        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.update_layout(
            title='3D Set',
            scene=dict( xaxis={"nticks": 20},
                        zaxis={"nticks": 20},
                        camera_eye={"x": 0, "y": -1, "z": 0.5},
                        aspectratio={"x": 1, "y": 1, "z": 0.6}
                        ))
        fig.show()
        
        
        
        
        
    


def plot_overlay_3Value(g, data1, data2, data3, plot_option):
    dims_plot = plot_option.dims_plot
    grid1, my_V1 = pre_plot(plot_option, g, data1)
    grid2, my_V2 = pre_plot(plot_option, g, data2)
    grid2, my_V3 = pre_plot(plot_option, g, data3)

    # my_V = np.zeros(my_V1.shape)
    # np.where(my_V1 != 0, 0, 1)
    # np.where(my_V2 != 0, 0, 2)

    # my_V = np.maximum(my_V1, my_V2)
    
    if len(dims_plot) == 2:
        # plot 2 2D sets
        # dim1, dim2 = dims_plot[0], dims_plot[1]
        complex_x = complex(0, grid1.pts_each_dim[dims_plot[0]])
        complex_y = complex(0, grid1.pts_each_dim[dims_plot[1]])
        mg_X, mg_Y = np.mgrid[grid1.min[0]:grid1.max[0]: complex_x, grid1.min[1]:grid1.max[1]: complex_y]
        
        # N = my_V1.shape[2]
        
        trace1 = go.Surface(
                    contours = {"z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white",}},
                    x=mg_X.flatten(),
                    y=mg_Y.flatten(),
                    z=my_V1.flatten(),
                    colorscale = plot_option.colorscale,
                    opacity = plot_option.opacity,
                    lighting = plot_option.lighting,
                    lightposition=plot_option.lightposition
                    ),
          
        trace2 = go.Surface(
                    contours = {"z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white",}},
                    x=mg_X.flatten(),
                    y=mg_Y.flatten(),
                    z=my_V2.flatten(),
                    colorscale = plot_option.colorscale,
                    opacity = plot_option.opacity,
                    lighting = plot_option.lighting,
                    lightposition=plot_option.lightposition
                    ),
        
        trace3 = go.Surface(
                    contours = {"z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white",}},
                    x=mg_X.flatten(),
                    y=mg_Y.flatten(),
                    z=my_V3.flatten(),
                    colorscale = plot_option.colorscale,
                    opacity = plot_option.opacity,
                    lighting = plot_option.lighting,
                    lightposition=plot_option.lightposition
                    ),
        
        fig = go.Figure()
        fig.add_trace(trace1)
        fig.add_trace(trace2)
        fig.add_trace(trace3)
        fig.update_layout(
            title='2D Value Function',
            scene=dict( xaxis={"nticks": 20},
                        zaxis={"nticks": 4},
                        camera_eye={"x": 0, "y": -1, "z": 0.5},
                        aspectratio={"x": 1, "y": 1, "z": 0.2}
                        ))
        fig.show()
        
        