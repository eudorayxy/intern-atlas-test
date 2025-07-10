import numpy as np
import time
import awkward as ak
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import MaxNLocator, AutoMinorLocator # for minor ticks


# 09/07/2025
# data_list consists of real data dict(s) that holds data for one variable,
# and mc dict(s) that holds data for totalWeight and another variable
# mainly this is to plot data that passed different selection cuts, or compare one final state with another for a single variable
# example: cutting on pt < 6 vs pt < 5, lep_pt[0] vs lep_pt[1]
def get_results_list(data_list, color_lists, label_list, variable):
    results_list = []

    if not (len(data_list) == len(color_lists) == len(label_list)):
        raise ValueError("Input lists for data dicts list, color lists, and label list must have the same length.")

    for data_dict, color_list, label in zip(data_list, color_lists, label_list):
        # data_dict = {'Data 2to4lep': {'lep_pt_0': <Array [39.4, 45.2, 33.8, 44.5, ..., 54.3, 35.6, 46.8] type='317512 * float32'>},
        # 'Signal $Z→μμ$': {'lep_pt_0': <Array [101, 75.4, 59.2, 183, ..., 34.1, 78.3, 30.4] type='484134 * float32'>,
                         # 'totalWeight': <Array [0.0388, 0.266, 0.616, ..., 0.00436, 0, 0.436] type='484134 * float64'>}}
        if not (len(data_dict.keys()) == len(color_list)):
            print(f'Number of data samples = {len(data_dict.keys())}, length of color list = {len(color_list)}')
            raise ValueError("Invalid size for inner color list.")
    
        for key, color in zip(data_dict, color_list):
            inner_dict = data_dict[key]
            new_dict = {'color' : color}

            # Count number of variable(s) in data
            available_variable_count = len(inner_dict)
            count = 0 # Count how many times input variable does not match the variable(s) during the loop
            for inner_key in inner_dict:
                if 'totalWeight' in inner_key:
                    new_dict['totalWeight'] = inner_dict[inner_key]

                if inner_key == variable:
                    new_inner_key = f'{label} {key} {variable}' # This will be the label of the plot
                    new_dict[new_inner_key] = inner_dict[variable]
                    break # Found matched variable and updated new_dict
                else:
                    count += 1
            # Variable not found in data
            if count == available_variable_count:
                raise ValueError(f"Variable {variable} not found for data {key} (label = {label})")
            
            results_list.append(new_dict)
    return results_list

def plot_errorbar(main_axes, selected_data, bin_edges, bin_centres, marker):
    color = selected_data['color']
        
    for key in selected_data:
        if key == 'color' or key == 'totalWeight':
            continue
                
        values = ak.to_numpy(selected_data[key])
        weights = ak.to_numpy(selected_data.get('totalWeight', np.ones_like(values)))
        
        data, _ = np.histogram(values, bins=bin_edges, weights=weights) # histogram the data
        
        data_err, _ = np.histogram(values, bins=bin_edges, weights=weights**2)
        data_err = np.sqrt(data_err)
        
        #data_errors = np.sqrt(data) # statistical error on the data
        # plot the data points
        main_axes.errorbar(x=bin_centres, y=data, yerr=data_err,
                            marker=marker, color=color, linestyle='none',
                            label=f'{key}')


def plot_multiple_runs( results_list,
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
    
    plt.figure(figsize=(12, 8))  # Create empty figure
    
    main_axes = plt.gca()

    for selected_data in results_list:
        # {'$p_t$[0] Run 1': {'color': 'b',
        # 'lep_pt_0': <Array [46.9, 42.8, 35.3, 34.4, ..., 14.2, 51.2, 36.9] type='5276419 * float32'>}} 
        plot_errorbar(main_axes, selected_data, bin_edges, bin_centres, marker)
           
        
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
    main_axes.set_ylabel(y_label, fontsize=label_fontsize,
                         y=1, horizontalalignment='right') 
    
    # draw the legend
    main_axes.legend(frameon=False, fontsize=legend_fontsize) # no box around the legend    
    
    main_axes.set_title(title, fontsize=label_fontsize)
    
    if logy:
        main_axes.set_yscale('log')

    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

    return main_axes
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------


# selected_data_list is a list of objects returned by get_data_pq
# Example: selected_data_list = {{'Data 2to4lep' : {'lep_pt_0' : ak.Array(...)}}, 
                                #{'Data 2to4lep' : {'lep_pt_1' : ak.Array(...)}},
                                #{'Data 2to4lep' : {'lep_pt_1' : ak.Array(...)}}}
# Example: color_list = ['r', 'g', 'b']
# Example: label_list = ['$p_t$[0] Run 1', '$p_t$[1] Run 1', '$p_t$[1] Run 2']
# Example return [{'$p_t$[0] Run 1' : {'color' : 'r', 'lep_pt_0' : ak.Array(...)}},
                # {'$p_t$[1] Run 1' : {'color' : 'g', 'lep_pt_1' : ak.Array(...)}},
                # {'$p_t$[1] Run 2' : {'color' : 'b', 'lep_pt_1' : ak.Array(...)}}]
def get_multiple_runs_real_data_list(selected_data_list, color_list, label_list):
    multiple_runs_data_list = []

    if not (len(selected_data_list) == len(color_list) == len(label_list)):
        raise ValueError("Input lists for data dicts, color list, and label list must have the same length.")

    for selected_data, color, label in zip(selected_data_list, color_list, label_list):
        # selected_data = {'Data 2to4lep': {'lep_pt_0': ak.Array(...)}} 
        # Get the inner dict {'lep_pt_0': ak.Array(...)}
        for _, inner_dict in selected_data.items():  # only one key-value pair
            data_dict = {'color': color}
            data_dict.update(inner_dict)
            multiple_runs_data_list.append({label : data_dict})
            
    return multiple_runs_data_list

def plot_main_errorbar(main_axes, selected_data, bin_edges, bin_centres, marker):
    for key, value_dict in selected_data.items(): # loop over samples

        color = value_dict['color']
        weights = ak.to_numpy(value_dict.get('totalWeight', np.ones_like(values)))
        
        for variable in value_dict:
            if variable == 'color' or variable == 'totalWeight':
                continue
                
            values = ak.to_numpy(value_dict[variable])

            data, _ = np.histogram(values, bins=bin_edges, weights=weights) # histogram the data
            
            data_err, _ = np.histogram(values, bins=bin_edges, weights=weights**2)
            data_err = np.sqrt(data_err)
            
            #data_errors = np.sqrt(data) # statistical error on the data
            # plot the data points
            main_axes.errorbar(x=bin_centres, y=data, yerr=data_err,
                                marker=marker, color=color, linestyle='none',
                                label=f'{key} {variable}')

def plot_multiple_runs_pq( multiple_runs_data_list,
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
    
    fig = plt.figure(figsize=(12, 8))  # Create empty figure
    
    main_axes = plt.gca()

    for selected_data in multiple_runs_data_list:
        # {'$p_t$[0] Run 1': {'color': 'b',
        # 'lep_pt_0': <Array [46.9, 42.8, 35.3, 34.4, ..., 14.2, 51.2, 36.9] type='5276419 * float32'>}} 
        plot_main_errorbar(main_axes, selected_data, bin_edges, bin_centres, marker)
           
        
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
    main_axes.set_ylabel(y_label, fontsize=label_fontsize,
                         y=1, horizontalalignment='right') 
    
    # draw the legend
    main_axes.legend(frameon=False, fontsize=legend_fontsize) # no box around the legend    
    
    main_axes.set_title(title, fontsize=label_fontsize)
    
    if logy:
        main_axes.set_yscale('log')

    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

    return fig, main_axes