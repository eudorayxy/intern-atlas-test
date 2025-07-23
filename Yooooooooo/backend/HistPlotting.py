import time
import awkward as ak
import numpy as np # # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import MaxNLocator, AutoMinorLocator # for minor ticks
import hist
from hist import Hist

def validate_xlabel(x_label_list, variable_list):
    variable_count = len(variable_list)
    x_label = []
    if isinstance(x_label_list, str): # If input is a string
        x_label = [x_label_list] * variable_count
    elif isinstance(x_label_list, (list, tuple)):
        if len(x_label_list) == 0: # If input is an empty list
            for variable in variable_list:
                x_label.append(variable)
        elif len(x_label_list) == variable_count:
            for i in x_label_list:
                x_label.append(str(i))
        else:
            raise ValueError("Input list/tuple for x axis label must have "
                             " the same length as variable list.")
    else:
         raise TypeError("Input for x axis label must be a str or a list/tuple with"
                         " the same length as variable list.")
    return x_label

def validate_ylabel(y_label_list, variable_count):
    y_label = []
    if isinstance(y_label_list, str): # If input is a string
        y_label = [y_label_list] * variable_count
    elif isinstance(y_label_list, (list, tuple)):
        if len(y_label_list) == 0: # If input is default or an empty string
            for i in range(variable_count):
                y_label.append('')
        elif len(y_label_list) == variable_count:
            for i in y_label_list:
                y_label.append(str(i))    
        else:
            raise ValueError("Input list/tuple for y axis label must have"
                             " the same length as variable list.")
    else:
        raise TypeError("Input for y axis label must be a str or a list/tuple with"
                        " the same length as variable list.")
    return y_label

def validate_title(title_list, variable_count):
    title = []
    if isinstance(title_list, str): # If input is a string
        title = [title_list] * variable_count
    elif isinstance(title_list, (list, tuple)):
        if len(title_list) == 0: # If input is default or an empty string
            for i in range(variable_count):
                title.append('')
        elif len(title_list) == variable_count:
            for i in title_list:
                title.append(str(i))    
        else:
            raise ValueError("Input list/tuple for title must have"
                             " the same length as variable list.")
    else:
        raise TypeError("Input for title must be a str or a list/tuple with"
                        " the same length as variable list.")
    return title

def validate_num_bins(num_bins_list, variable_count):
    num_bins = []
    
    if isinstance(num_bins_list, int):
        num_bins = [num_bins_list] * variable_count
    elif isinstance(num_bins_list, (list, tuple)):
        if len(num_bins_list) == variable_count:
            try:
                for i in num_bins_list:
                    num_bins.append(int(i))
            except TypeError:
                print("Input for number of bins must be an int or"
                      " a list/tuple of int with the same length as variable "
                      "list.")
                raise
            except Exception as e:
                print(f"Unexpected exception : {e}")
                raise
        else:
            raise ValueError("Invalid list/tuple length for num_bins_list.")
    else: # Invalid input type
        raise TypeError("Input for number of bins must be an int or a"
                        " list/tuple of int with the same length as variable list.")
                
    return num_bins
            
    
def validate_xmin_xmax(xmin_xmax_list, variable_count):
    if isinstance(xmin_xmax_list, str):
        raise TypeError("Input for xmin and xmax must not be a str.")

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
            if not isinstance(pair, (list, tuple)):
                raise TypeError("Each element must be a tuple/list of two numbers (xmin, xmax).")

            if len(pair) != 2:
                raise ValueError("Each element must be a tuple/list of two numbers (xmin, xmax).")
                
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

def stacked_histogram(all_data, color_list, variable, xmin, xmax, num_bins, main_axes, marker, show_back_unc):
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

    bin_edges = np.linspace(xmin, xmax, num_bins + 1)
    bin_centres = (bin_edges[:-1] # left edges
                    + bin_edges[1:] # right edges
                      ) / 2
    widths = (bin_edges[1:] # left edges
                - bin_edges[:-1]) # right edges

    hists = []
    text = []
    index = 1
    
    for (key, value), color in zip(all_data.items(), color_list): # loop over samples
        # Validate the input variable
        if variable not in value.keys():
            raise ValueError(f"Variable '{variable}' not found in sample '{key}'. Available varaible(s):"
                           f"{value.keys()}")
            
        if 'Data' in key:
            hist_data = Hist.new.Reg(num_bins, xmin, xmax, name=key).Double()
            # Fill values, replacing any None with nan first
            hist_data.fill(ak.to_numpy(ak.fill_none(value[variable], np.nan)))

            text.append(f'({index}) {key}: Sum (value = {sum(hist_data.view(flow=False)):.3e}),')
            text.append(f'Underflow = {hist_data.view()[0]:.3e}, Overflow = {hist_data.view()[-1]:.3e}\n')
            index += 1
        
            data_x = hist_data.view(flow=False)
            data_x_errors = np.sqrt(data_x) # statistical error on the data
            hists.append(hist_data)
            
            # Plot the data points
            main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                marker=marker, color=color, linestyle='none',
                                label=key) 
        elif 'Signal' in key:
            signal_x.append(ak.to_numpy(ak.fill_none(value[variable], np.nan))) # histogram the signal
            signal_weights.append(ak.to_numpy(value['totalWeight'])) # get the weights of the signal events
            signal_colors.append(color) # get the colour for the signal bar
            signal_labels.append(key)
            signal_input = True # MC signal present
        else:
            background_x.append(ak.to_numpy(ak.fill_none(value[variable], np.nan))) # append to the list of MC histogram entries
            background_weights.append(ak.to_numpy(value['totalWeight'])) # append to the list of MC weights
            background_colors.append(color) # append to the list of MC bar colors
            background_labels.append(key) # append to the list of MC legend labels
            background_input = True # Background signal present
        
     
    if background_input: # Background signal present
        background_hists = []
        for back_data, back_weight, label in zip(background_x, background_weights, background_labels):
            back_hist = Hist.new.Reg(num_bins, xmin, xmax, name=label).Weight()
            back_hist.fill(back_data, weight=back_weight)
            text.append(f'({index}) {label}: Weighted Sum (value = {back_hist.sum().value:.3e}, '
                        f'variance = {back_hist.sum().variance:.3e}),')
            text.append(f'Underflow = {back_hist.view()[0].value:.3e}, Overflow = {back_hist.view()[-1].value:.3e}')
            index += 1
            background_hists.append(back_hist)
            hists.append(back_hist)

        # Total count of background data and variance
        back_stacked_counts = sum(h.view(flow=False).value for h in background_hists)
        variances = sum(h.view(flow=False).variance for h in background_hists)
        stat_err = np.sqrt(variances) # Statistical uncertainty

        # Plot the MC bars
        bottom = np.zeros_like(back_stacked_counts)
        for h, color, label in zip(background_hists, background_colors, background_labels):
            counts = h.view(flow=False).value
            plt.bar(x=bin_centres, height=counts, width=widths, bottom=bottom, color=color, label=label, align='center')
            bottom += counts

        if show_back_unc:
            # Plot the statistical uncertainty
            plt.bar(x=bin_centres, height=stat_err*2, width=widths, bottom=bottom-stat_err, color='none', hatch="////", label='Stat. Unc.', align='center')
        
    else:
        back_stacked_counts = np.zeros(len(bin_centres))
    
    if signal_input: # MC signal present
        signal_hists = []
        for data, weight, label in zip(signal_x, signal_weights, signal_labels):
            signal_hist = Hist.new.Reg(num_bins, xmin, xmax, name=label).Weight()
            signal_hist.fill(data, weight=weight)
            text.append(f'({index}) {label}: Weighted Sum (value = {signal_hist.sum().value:.3e}, '
                        f'variance = {signal_hist.sum().variance:.3e}),')
            text.append(f'Underflow = {signal_hist.view()[0].value:.3e}, '
                        f'Overflow = {signal_hist.view()[-1].value:.3e}')
            index += 1
            signal_hists.append(signal_hist)
            hists.append(signal_hist)
            
        # Total count of signal data
        stacked_counts = sum(h.view(flow=False).value for h in signal_hists)

        # Plot the bars
        bottom = back_stacked_counts
        for h, color, label in zip(signal_hists, signal_colors, signal_labels):
            counts = h.view(flow=False).value
            plt.bar(x=bin_centres, height=counts, width=widths, bottom=bottom, color=color,
                    label=label, align='center')
            bottom += counts

    return hists, text

# This function aims to plot one or more variables from a single run (constant selection cut)
def plot_histogram_hist(
        data,
        variable_list,
        color_list,
        xmin_xmax_list,
        num_bins_list,
        x_label_list,
        # Optional arguments start from here
        y_label_list=[], # Str or list of str for y axis label
        logy=False, # Whether to set the y axis as log scale
        title_list=[], # Str or list of str for title
        marker='o', # Marker type
        title_fontsize=17, # Fontsize for title
        label_fontsize=17, # Fontsize for x and y axes
        legend_fontsize=17, # Fontsize for legend
        tick_labelsize=15, # Fontsize for x and y axes ticks
        text_fontsize=14, # Fontsize for text that shows histogram info
        fig_size=(12, 8), # Figure size
        show_text=False, # Bool - whether to show the text that displays histogram info
        show_back_unc=True, # Bool - whether to show the background uncertainty
        return_fig_hist=True # Bool - whether to return the figure(s) list and histogram object(s) list
    ):

    if isinstance(color_list, str):
        raise TypeError('Input for colors must be a list.')

    # Check if all input lists are the same length
    if len(data.keys()) != len(color_list):
        raise ValueError("Input lists for sample keys and colors must have the same length.") 

    # Validate fig_size
    if (not isinstance(fig_size, tuple)
        or not all(isinstance(value, (int, float)) for value in fig_size)):
        raise ValueError("fig_size must be a tuple of two numbers.")
    if len(fig_size) != 2 or not all(value > 0 for value in fig_size):
        raise ValueError("fig_size must be a tuple of two positive numbers.")

    time_start = time.time()

    # Validate variable_list
    if isinstance(variable_list, (list, tuple)):
        variable_count = len(variable_list)
    else:
        variable_count = 1 # It is a str
        variable_list = [variable_list] * variable_count

    # Validate x label input
    # If x_label_list is one string, make it into an array corresponding to the length of other input list
    x_label_list = validate_xlabel(x_label_list, variable_list)

    # Validate ylabel input
    # If y_label_list is default (empty list) or user input is a string or user_input list length not matched
    y_label_list = validate_ylabel(y_label_list, variable_count)
    
    # Validate title
    title_list = validate_title(title_list, variable_count)
    
    # Validate number of bins input
    num_bins_list = validate_num_bins(num_bins_list, variable_count)
        
    # Validate xmin xmax input                
    xmin_xmax_list = validate_xmin_xmax(xmin_xmax_list, variable_count)

    if return_fig_hist:
        fig_list = []
        hists_list = []
                
    for  axes_count, (variable, x_label, y_label,
         xmin_xmax, num_bins, title) in enumerate(zip(variable_list, x_label_list,
                                                y_label_list, xmin_xmax_list,
                                                num_bins_list, title_list)):
        
        fig, main_axes = plt.subplots(figsize=fig_size)
        
        xmin, xmax = xmin_xmax
        hists, text = stacked_histogram(data, color_list, variable, xmin, xmax, num_bins, main_axes, marker, show_back_unc)
        # set the x-limit of the main axes
        #main_axes.set_xlim(left=xmin-step_size*5, right=xmax) 

        if show_text:
             for i, line in enumerate(text):
                 main_axes.text(-0.05, -0.15 - i * 0.05, line, ha='left', va='top', transform=main_axes.transAxes, fontsize=text_fontsize)
            
        # separation of x axis minor ticks
        main_axes.xaxis.set_minor_locator(AutoMinorLocator()) 
        
        # set the axis tick parameters for the main axes
        main_axes.tick_params(which='both', # ticks on both x and y axes
                              direction='in', # Put ticks inside and outside the axes
                              labelsize=tick_labelsize, # Label size
                              top=True, # draw ticks on the top axis
                              right=True) # draw ticks on right axis
    
    
        # x-axis label
        main_axes.set_xlabel(x_label, fontsize=label_fontsize,
                             x=1, horizontalalignment='right' )

        if y_label == '':
            y_label = 'Events'
            
        # write y-axis label for main axes
        main_axes.set_ylabel(y_label,
                             fontsize=label_fontsize,
                             y=1, horizontalalignment='right') 
        
        # draw the legend
        main_axes.legend(frameon=False, fontsize=legend_fontsize) # no box around the legend    
        
        main_axes.set_title(title, fontsize=title_fontsize)
        
        if logy:
            main_axes.set_yscale('log')
    
        elapsed_time = time.time() - time_start 
        print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

        # Show the plot
        plt.tight_layout()
        plt.show()

        if return_fig_hist:
            hists_list.append(hists)
            fig_list.append(fig)

    if return_fig_hist:
        return fig_list, hists_list


def histogram_2d(data, variable, num_bins, min_max, label=['',''],
                 label_fontsize=12, tick_labelsize=10, title_fontsize=13, title='', cbar_label=''):
    # Validate variable
    if not isinstance(variable, (list, tuple)):
        raise TypeError("variable must be a list or tuple of two str.")
        
    # Validate the length of input
    if (
        len(data) != 2 or len(variable) != 2
        or len(num_bins) != 2 or len(min_max) != 2
        or len(label) != 2
        ):
        raise ValueError('All inputs must have a length of two.')

    # Validate the number of bins
    if not all(isinstance(i, int) for i in num_bins): # Must be an int
        raise TypeError('num_bins have to be a tuple or list of numbers.')
    if not all(i > 0 for i in num_bins): # Must be positive
        raise ValueError('num_bins have to be a tuple or list of two positive numbers.')

    # Validate the min and max points for x and y
    for pair in min_max:
        if not all(isinstance(i, (int, float)) for i in pair):
            raise TypeError('min_max must be a pair of tuple/list of two numbers.')
        if not (pair[1] - pair[0]) > 0:
            raise ValueError('The second number in each tuple/list of min_max must be larger than the first.')

    # Label of axes
    # Convert tuple input into list for assignment
    if isinstance(label, tuple):
        label = [label[0], label[1]]
    # Assign variable to label if label for x or/and y is an empty string
    for i, j in enumerate(label):
        if j == '':
            label[i] = variable[i]
        else:
            label[i] = str(j)

    # Extract values for x and y from inputs
    data_x, data_y = data
    variable_x, variable_y = variable
    num_bins_x, num_bins_y = num_bins
    (xmin, xmax), (ymin, ymax) = min_max
    label_x, label_y = label

    # Validate the variable
    if variable_x not in data_x:
        raise ValueError(f'Variable {variable_x} not found in the x data. Available variables: {list(data_x.keys())}')
    if variable_y not in data_y:
        raise ValueError(f'Variable {variable_y} not found in the y data. Available variables: {list(data_y.keys())}')

    # Histogram
    h = Hist(
        hist.axis.Regular(num_bins_x, xmin, xmax, name=variable_x, label=label_x, flow=False),
        hist.axis.Regular(num_bins_y, ymin, ymax, name=variable_y, label=label_y, flow=False)
    )
    h.fill(ak.to_numpy(data_x[variable_x]), ak.to_numpy(data_y[variable_y]))

    # Plot 2D histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    values, x_bin_edges, y_bin_edges = h.to_numpy()
    mesh = ax.pcolormesh(x_bin_edges, y_bin_edges, values.T, cmap="viridis")
    
    ax.set_xlabel(label_x, fontsize=label_fontsize)
    ax.set_ylabel(label_y, fontsize=label_fontsize)
    ax.tick_params(which='both', labelsize=tick_labelsize)
    ax.set_title(str(title), fontsize=title_fontsize)
    
    # Create colorbar and adjust its label and tick label sizes
    cbar = fig.colorbar(mesh)
    cbar.set_label(cbar_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)
    plt.show()
    return fig, h
    