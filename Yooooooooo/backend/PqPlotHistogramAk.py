# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:42:38 2025

@author: c01712ey
"""
import time
import awkward as ak
import numpy as np # # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import MaxNLocator, AutoMinorLocator # for minor ticks

def stacked_histogram(all_data, color_list, variable, bin_centres, bin_edges, step_size, main_axes, marker):
    background_x = [] # define list to hold the MC histogram entries
    background_weights = [] # define list to hold the MC weights
    background_colors = [] # define list to hold the colors of the MC bars
    background_labels = [] # define list to hold the legend labels of the MC bar
    
    signal_x = []
    signal_weights = []
    signal_colors = []
    signal_labels = []
    
    signal_input = False
    background_input = False
    
    for (key, value), color in zip(all_data.items(), color_list): # loop over samples

        # Validate the input variable
        if variable not in value.keys():
            raise KeyError(f"Variable '{variable}' not found in sample '{key}'. Available varaible(s):"
                           f"{value.keys()}")
            
        if 'Data' in key:
            data_x, _ = np.histogram(ak.to_numpy(value[variable]), 
                                    bins=bin_edges) # histogram the data
            data_x_errors = np.sqrt(data_x) # statistical error on the data
            # plot the data points
            main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                marker=marker, color=color, linestyle='none',
                                label=key) 
        elif 'Signal' in key:
            signal_x.append(ak.to_numpy(value[variable])) # histogram the signal
            signal_weights.append(ak.to_numpy(value['totalWeight'])) # get the weights of the signal events
            signal_colors.append(color) # get the colour for the signal bar
            signal_labels.append(key)
            signal_input = True
        else:
            background_x.append(ak.to_numpy(value[variable]) ) # append to the list of MC histogram entries
            background_weights.append(ak.to_numpy(value['totalWeight'])) # append to the list of MC weights
            background_colors.append(color ) # append to the list of MC bar colors
            background_labels.append(key) # append to the list of MC legend labels
            background_input = True
     
    if background_input:
        # Plot the MC bars
        background_heights = main_axes.hist(background_x, 
                                            bins=bin_edges, 
                                            weights=background_weights,
                                            stacked=True, 
                                            color=background_colors,
                                            label=background_labels)
        
        # Stores the total stacked y-values of the background background histogram at each bin
        background_x_tot = background_heights[0][-1] # stacked background background y-axis value
        
        # Calculate background statistical uncertainty: sqrt(sum w^2)
        background_x_err = np.sqrt(np.histogram(np.hstack(background_x),
                                                bins=bin_edges,
                                                weights=np.hstack(background_weights)**2)[0])
        # Plot the statistical uncertainty
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

def validate_xlabel(x_label_list, variable_list):
    variable_count = len(variable_list)
    x_label = []
    if isinstance(x_label_list, str): # If input is a string
        for i in range(variable_count):
            x_label.append(x_label_list)
    elif len(x_label_list) == 0: # If input is an empty list
        for variable in variable_list:
            x_label.append(variable)
    elif len(x_label_list) == variable_count:
        for i in x_label_list:
            x_label.append(str(i))
    else:
        raise ValueError("Input for x axis label must be a str or a list with"
                         " the same length as variable list.")
    return x_label

def validate_ylabel(y_label_list, variable_count):
    y_label = []
    if isinstance(y_label_list, str): # If input is a string
        for i in range(variable_count):
            y_label.append(y_label_list)
    elif len(y_label_list) == 0: # If input is default or an empty string
        for i in range(variable_count):
            y_label.append('')
    elif len(y_label_list) == variable_count:
        for i in y_label_list:
            y_label.append(str(i))    
    else:
        raise ValueError("Input for y axis label must be a str or a list with"
                         " the same length as variable list.")
    return y_label

def validate_title(title_list, variable_count):
    title = []
    if isinstance(title_list, str): # If input is a string
        for i in range(variable_count):
            title.append(title_list)
    elif len(title_list) == 0: # If input is default or an empty string
        for i in range(variable_count):
            title.append('')
    elif len(title_list) == variable_count:
        for i in title_list:
            title.append(str(i))    
    else:
        raise ValueError("Input for title must be a str or a list with"
                         " the same length as variable list.")
    return title

def validate_stepsize(step_size_list, variable_count):
    step_size = []
    try:
        # Only an int or float is a valid input
        step_size_list = float(step_size_list)
        for i in range(variable_count):
            step_size.append(step_size_list)
        
    except Exception as e:
        if isinstance(step_size_list, str):
            raise TypeError("Input for step size must be an int or float or a"
                            " list with the same length as variable list.") from e
        
        elif len(step_size_list) == variable_count:
            try:
                for i in step_size_list:
                    step_size.append(float(i))
            except Exception as e:
                raise TypeError("Input for step size must be a float or int or"
                                " a list with the same length as variable "
                                "list.") from e
        else: # Invalid input list length
            raise ValueError("Input for step size must be an int or float or a"
                             " list with the same length as variable list.")
                
    return step_size
            
    

def validate_xmin_xmax(xmin_xmax_list, variable_count):
    if isinstance(xmin_xmax_list, str):
        raise ValueError("Input for xmin and xmax must not be a str.")

    xmin_xmax = [] # Store validated input

    # Case 1: Single (xmin, xmax) tuple/list for all variables
    if (isinstance(xmin_xmax_list, (list, tuple)) # Only accept a tuple or list, avoid str input
        and len(xmin_xmax_list) == 2 # Only accept (a, b) or [a, b]
        # where a and b are int or float
        and all(isinstance(value, (int, float)) for value in xmin_xmax_list)
       ): 
        xmin, xmax = xmin_xmax_list
        # Raise error if xmax is smaller than or equal to xmin
        if xmax <= xmin:
            raise ValueError(f"Input xmin = {xmin}, xmax = {xmax} : xmax must be greater than xmin.")
        # Make xmin_xmax a list with the same length as variable list
        xmin_xmax = [(xmin, xmax)] * variable_count
        return xmin_xmax

    # Case 2: List of (xmin, xmax) pairs for each variable.
    if (isinstance(xmin_xmax_list, (list, tuple)) # Avoid str input
        and len(xmin_xmax_list) == variable_count # Number of pairs must match number of variables
       ):
        for pair in xmin_xmax_list:
            # Raise error if each object in xmin_xmax_list is not a tuple or list of 2 numbers
            if not isinstance(pair, (list, tuple)) and len(pair) != 2:
                raise TypeError("Each element must be a tuple/list of two numbers (xmin, xmax).")
                
            xmin, xmax = pair
            # Only accept (a, b) or [a, b] where a and b are int or float
            if not all(isinstance(value, (int, float)) for value in (xmin, xmax)):
                raise TypeError(f"Input xmin = {xmin}, xmax = {xmax} : xmin and xmax must be int or float.")
            # Raise error if xmax is smaller than or equal to xmin
            if xmax <= xmin:
                raise ValueError(f"Input xmin = {xmin}, xmax = {xmax} : xmax must be greater than xmin.")
            # Update with validated input    
            xmin_xmax.append((xmin, xmax))
        return xmin_xmax

    # If none of the above formats match
    raise ValueError(
        "Invalid format for xmin_xmax. Must be either:\n"
        "1. A tuple/list of two numbers: (xmin, xmax)\n"
        "2. A list of (xmin, xmax) tuples/lists of same length as variable count. "
        f"Number of input variables = {variable_count}."
    )       

# This function aims to plot one or more variables from a single run (constant selection cut)
def plot_histogram_new(all_data,
                   variable_list,
                   color_list,
                   xmin_xmax_list,
                   step_size_list,
                   x_label_list,
                   y_label_list=[],
                   logy=False,
                   title_list=[],
                   marker='o',
                   title_fontsize=13,
                   label_fontsize=13,
                   legend_fontsize=13):

    # Check if all input lists are the same length
    if len(all_data.keys()) != len(color_list) or isinstance(color_list, str):
        raise ValueError("Input lists for sample keys and colors must have the same length.")  

    time_start = time.time()
    
    variable_count = len(variable_list)

    # Validate x label input
    # If x_label_list is one string, make it into an array corresponding to the length of other input list
    x_label_list = validate_xlabel(x_label_list, variable_list)

    # Validate ylabel input
    # If y_label_list is default (empty list) or user input is a string or user_input list length not matched
    y_label_list = validate_ylabel(y_label_list, variable_count)
    
    # Validate title
    title_list = validate_title(title_list, variable_count)
    
    # Validate step size input
    step_size_list = validate_stepsize(step_size_list, variable_count)
        
    # Validate xmin xmax input                
    xmin_xmax_list = validate_xmin_xmax(xmin_xmax_list, variable_count)

    # Handle variable given as a str
    if not isinstance(variable_list, (list, tuple)):
        variable_list = [variable_list]
                
    for (variable, x_label, y_label,
         xmin_xmax, step_size, title) in zip(variable_list, x_label_list,
                                                y_label_list, xmin_xmax_list,
                                                step_size_list, title_list):
    
        xmin, xmax = xmin_xmax

        if step_size >= abs(xmax - xmin):
            raise ValueError("Step size needs to be smaller than the (xmax - xmin).")

        bin_edges = np.arange(start=xmin, # The interval includes this value
                          stop=xmax+step_size, # The interval doesn't include this value
                          step=step_size) # Spacing between values
        bin_centres = np.arange(start=xmin+step_size/2,
                            stop=xmax+step_size/2,
                            step=step_size)
        
        fig = plt.figure(figsize=(12, 8))  # Create empty figure
        
        main_axes = plt.gca()
        
        stacked_histogram(all_data, color_list, variable, bin_centres,
                          bin_edges, step_size, main_axes, marker)
            
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

        if y_label == '':
            y_label = f'Events / {step_size}'
            
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


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
def get_multiple_runs_data_list(all_data_list, color_lists, label_list):
    multiple_runs_data_list = []

    if not (len(all_data_list) == len(color_lists) == len(label_list)):
        raise ValueError("Input lists for data dicts, color lists and label list must have the same length.")

    for all_data, color_list, label in zip(all_data_list, color_lists, label_list):
        # Validate the length of input lists
        if not (len(all_data.keys()) == len(color_list)):
            raise ValueError("Input lists for sample keys and colors must have the same length.")

        # Create new key for each run using the input label list
        for key in all_data:
            all_data[f"{key} {label}"] = all_data[key]
            
        for key, color in zip(all_data, color_list):
            all_data[key]['color'] = color

        multiple_runs_data_list.append(all_data)
    return multiple_runs_data_list


def plot_multiple_runs_pq( all_data_list,
                           variable,
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
                           legend_fontsize=13):

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
    
    background_x = [] # define list to hold the MC histogram entries
    background_weights = [] # define list to hold the MC weights
    background_colors = [] # define list to hold the colors of the MC bars
    background_labels = [] # define list to hold the legend labels of the MC bar
    
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
        if variable not in value.keys():
            raise KeyError(f"Variable '{variable}' not found in sample '{key}'. Available varaible(s):"
                           f"{value.keys()}")
            
        if 'Data' in key:
            data_x, _ = np.histogram(ak.to_numpy(value[variable]), 
                                    bins=bin_edges) # histogram the data
            data_x_errors = np.sqrt(data_x) # statistical error on the data
            # plot the data points
            main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                marker=marker, color=color, linestyle='none',
                                label=key) 
        elif 'Signal' in key:
            signal_x.append(ak.to_numpy(value[variable])) # histogram the signal
            signal_weights.append(ak.to_numpy(value['totalWeight'])) # get the weights of the signal events
            signal_colors.append(color) # get the colour for the signal bar
            signal_labels.append(key)
            signal_input = True
        else:
            background_x.append(ak.to_numpy(value[variable]) ) # append to the list of MC histogram entries
            background_weights.append(ak.to_numpy(value['totalWeight'])) # append to the list of MC weights
            background_colors.append(color ) # append to the list of MC bar colors
            background_labels.append(key) # append to the list of MC legend labels
            background_input = True
     
    if background_input:
        # plot the MC bars
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