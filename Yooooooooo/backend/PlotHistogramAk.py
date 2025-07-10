# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:42:38 2025

@author: c01712ey
"""
import time
import awkward as ak
import numpy as np # # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import MaxNLocator, AutoMinorLocator # for minor ticks

def plot_histogram(all_data,
                   variable,
                   color_list,
                   xmin,
                   xmax,
                   step_size,
                   x_label,
                   y_label='Events',
                   logy=False,
                   title='',
                   marker='o',
                   title_fontsize=13,
                   label_fontsize=13,
                   legend_fontsize=13,
                   index=0,
                   scalar_variable=True):

    # Check if all input lists are the same length
    if not (len(all_data.keys()) == len(color_list)):
        raise ValueError("Input lists for sample keys and colors must have the same length.")

    if xmin >= xmax: 
        raise ValueError("xmax needs to be larger than xmin.")
    if step_size >= abs(xmax - xmin):
        raise ValueError("Step size needs to be smaller than the (xmax - xmin).")
        
    time_start = time.time()
    
    bin_edges = np.arange(start=xmin, # The interval includes this value
                          stop=xmax+step_size, # The interval doesn't include this value    
                          step=step_size) # Spacing between values
    bin_centres = np.arange(start=xmin+step_size/2,
                            stop=xmax+step_size/2,
                            step=step_size)
    
    background_x = [] # define list to hold the Monte Carlo histogram entries
    background_weights = [] # define list to hold the Monte Carlo weights
    background_colors = [] # define list to hold the colors of the Monte Carlo bars
    background_labels = [] # define list to hold the legend labels of the Monte Carlo bar
    
    signal_x = []
    signal_weights = []
    signal_colors = []
    signal_labels = []
    
    signal_input = False
    background_input = False
    
    fig = plt.figure(figsize=(12, 8))  # Create empty figure
    
    main_axes = plt.gca()

    for (key, value), color in zip(all_data.items(), color_list): # loop over samples

        # Validate the input variable
        if variable not in value['results'].fields:
            raise KeyError(f"Variable '{variable}' not found in sample '{key}'. Available varaible(s):"
                           f"{value['results'].fields}")
        
        if scalar_variable:
            data = value['results'][variable]
        else: 
            data = value['results'][variable][:, index]
            
        if 'Data' in key:
            data_x, _ = np.histogram(ak.to_numpy(data), 
                                    bins=bin_edges) # histogram the data
            data_x_errors = np.sqrt(data_x) # statistical error on the data
            # plot the data points
            main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                marker=marker, color=color, linestyle='none',
                                label=key) 
        elif 'Signal' in key:
            signal_x.append(ak.to_numpy(data)) # histogram the signal
            signal_weights.append(ak.to_numpy(value['results']['totalWeight'])) # get the weights of the signal events
            signal_colors.append(color) # get the colour for the signal bar
            signal_labels.append(key)
            signal_input = True
        else:
            background_x.append(ak.to_numpy(data)) # append to the list of Monte Carlo histogram entries
            background_weights.append(ak.to_numpy(value['results']['totalWeight'])) # append to the list of Monte Carlo weights
            background_colors.append(color) # append to the list of Monte Carlo bar colors
            background_labels.append(key) # append to the list of Monte Carlo legend labels
            background_input = True
     
    if background_input:
        # plot the Monte Carlo bars
        background_heights = main_axes.hist(background_x, 
                                            bins=bin_edges, 
                                            weights=background_weights,
                                            stacked=True, 
                                            color=background_colors,
                                            label=background_labels)
        
        # Stores the total stacked y-values of the background background histogram at each bin
        background_x_tot = background_heights[0][-1] # stacked background background y-axis value
        
        # calculate background statistical uncertainty: sqrt(sum w^2)
        background_x_err = np.sqrt(np.histogram(np.hstack(background_x),
                                                bins=bin_edges,
                                                weights=np.hstack(background_weights)**2)[0])
        # plot the statistical uncertainty
        main_axes.bar(bin_centres, # x
                      2 * background_x_err, # heights
                      alpha=0.5, # half transparency
                      bottom=background_x_tot-background_x_err, color='none', 
                      hatch="////", width=step_size, label='Stat. Unc.' )
    else:
        background_x_tot = np.zeros(len(bin_centres))
    
    if signal_input:
        # plot the signal bar
        main_axes.hist(signal_x,
                       bins=bin_edges,
                       bottom=background_x_tot,
                       stacked=True,
                       weights=signal_weights,
                       color=signal_colors,
                       label=signal_labels)
        
    # set the x-limit of the main axes
    main_axes.set_xlim(left=xmin-step_size*5, right=xmax) 

    
    # separation of x axis minor ticks
    main_axes.xaxis.set_minor_locator(AutoMinorLocator()) 
    
    # set the axis tick parameters for the main axes
    main_axes.tick_params(which='both', # ticks on both x and y axes
                          direction='in', # Put ticks inside and outside the axes
                          labelsize=13, # Label size
                          top=True, # draw ticks on the top axis
                          right=True) # draw ticks on right axis


    # x-axis label
    main_axes.set_xlabel(x_label, fontsize=label_fontsize,
                         x=1, horizontalalignment='right' )
    
    # write y-axis label for main axes
    main_axes.set_ylabel(y_label,
                         fontsize=label_fontsize,
                         y=1, horizontalalignment='right') 
    
    # draw the legend
    main_axes.legend(frameon=False, fontsize=legend_fontsize) # no box around the legend    
    
    main_axes.set_title(title, fontsize=label_fontsize)
    
    if logy:
        main_axes.set_yscale('log')

    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

        
    return fig, main_axes

# def plot_histogram(all_data, variable, xmin, xmax, step_size):
    
#     bin_edges = np.arange(start=xmin, # The interval includes this value
#                           stop=xmax+step_size, # The interval doesn't include this value
#                           step=step_size) # Spacing between values
#     bin_centres = np.arange(start=xmin+step_size/2,
#                             stop=xmax+step_size/2,
#                             step=step_size)
    
#     background_x = [] # define list to hold the Monte Carlo histogram entries
#     background_weights = [] # define list to hold the Monte Carlo weights
#     background_colors = [] # define list to hold the colors of the Monte Carlo bars
#     background_labels = [] # define list to hold the legend labels of the Monte Carlo bar
    
#     signal_x = []
#     signal_weights = []
#     signal_colors = []
#     signal_labels = []
    
#     signal_input = False
#     background_input = False
    
#     fig, main_axes = plt.subplots(figsize=(12, 8))
    

#     for key, value in all_data.items(): # loop over samples
#         if 'Data' in key:
#             data_x, _ = np.histogram(ak.to_numpy(value['results'][variable]), 
#                                     bins=bin_edges) # histogram the data
#             data_x_errors = np.sqrt(data_x) # statistical error on the data
#             # plot the data points
#             main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
#                                 marker='.', color=value['color'], linestyle='none',
#                                 label=key) 
#         elif 'Signal' in key:
#             signal_x.append(ak.to_numpy(value['results'][variable])) # histogram the signal
#             signal_weights.append(ak.to_numpy(value['results']['totalWeight'])) # get the weights of the signal events
#             signal_colors.append(value['color']) # get the colour for the signal bar
#             signal_labels.append(key)
#             signal_input = True
#         else:
#             background_x.append(ak.to_numpy(value['results'][variable]) ) # append to the list of Monte Carlo histogram entries
#             background_weights.append(ak.to_numpy(value['results']['totalWeight'])) # append to the list of Monte Carlo weights
#             background_colors.append(value['color'] ) # append to the list of Monte Carlo bar colors
#             background_labels.append(key) # append to the list of Monte Carlo legend labels
#             background_input = True
     
#     if background_input:
#         # plot the Monte Carlo bars
#         background_heights = main_axes.hist(background_x, 
#                                             bins=bin_edges, 
#                                             weights=background_weights,
#                                             stacked=True, 
#                                             color=background_colors,
#                                             label=background_labels)
        
#         # Stores the total stacked y-values of the background background histogram at each bin
#         background_x_tot = background_heights[0][-1] # stacked background background y-axis value
        
#         # calculate background statistical uncertainty: sqrt(sum w^2)
#         background_x_err = np.sqrt(np.histogram(np.hstack(background_x),
#                                                 bins=bin_edges,
#                                                 weights=np.hstack(background_weights)**2)[0])
#         # plot the statistical uncertainty
#         main_axes.bar(bin_centres, # x
#                       2 * background_x_err, # heights
#                       alpha=0.5, # half transparency
#                       bottom=background_x_tot-background_x_err, color='none', 
#                       hatch="////", width=step_size, label='Stat. Unc.' )
#     else:
#         background_x_tot = np.zeros(len(bin_centres))
    
#     if signal_input:
#         # plot the signal bar
#         main_axes.hist(signal_x,
#                        bins=bin_edges,
#                        bottom=background_x_tot,
#                        stacked=True,
#                        weights=signal_weights,
#                        color=signal_colors,
#                        label=signal_labels)
        
#     # set the x-limit of the main axes
#     main_axes.set_xlim(left=xmin-step_size*5, right=xmax) 

    
#     # separation of x axis minor ticks
#     main_axes.xaxis.set_minor_locator(AutoMinorLocator()) 
    
#     # set the axis tick parameters for the main axes
#     main_axes.tick_params(which='both', # ticks on both x and y axes
#                           direction='in', # Put ticks inside and outside the axes
#                           labelsize=13, # Label size
#                           top=True, # draw ticks on the top axis
#                           right=True) # draw ticks on right axis


#     # x-axis label
#     main_axes.set_xlabel(variable + ' / GeV', fontsize=13,
#                          x=1, horizontalalignment='right' )
    
#     # write y-axis label for main axes
#     main_axes.set_ylabel('Events / ' + str(step_size) + ' GeV', fontsize=13,
#                          y=1, horizontalalignment='right') 
    
#     # draw the legend
#     main_axes.legend(frameon=False, fontsize=16) # no box around the legend    
        
#     return fig, main_axes


