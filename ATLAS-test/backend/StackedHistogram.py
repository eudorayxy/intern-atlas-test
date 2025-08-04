import re
import time
import awkward as ak
import numpy as np # # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
from matplotlib.ticker import AutoMinorLocator # for minor ticks
import hist
from hist import Hist 

def get_variable_data(variable, key, value, valid_var):
    
    if variable in valid_var:
        variable_data = value[variable]
        type_str = str(ak.type(variable_data))
        if "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str):
            raise ValueError(f'Invalid input variable format : {variable}. '
                             f'Expect "{variable}[int]".')
    else:
        if '[' in variable and ']' in variable:
            
            base_var = variable.split('[')[0]
            idx_pos_start = variable.find('[') + 1 # Index starting position in the str
            idx_pos_end = variable.find(']')
            index = variable[idx_pos_start : idx_pos_end]

            try:
                index = int(index)
            except Exception as e:
                print(f'Invalid input variable format : "{variable}". '
                      f'Expect "variable" or "variable[int]".\nError: {e}')

            variable_data = value[base_var]

            type_str = str(ak.type(value[base_var]))
            is_nested = "var *" in type_str or re.search(r"\*\s*\d+\s*\*", type_str)
            if is_nested:
                num = ak.num(variable_data)
                max_num = ak.max(num)
                if index >= max_num:
                   raise IndexError(f'Invalid index for input variable "{variable}". '
                                    f'Input index should be less than {max_num}.')
                if not ak.all(num >= index + 1):
                    data = ak.pad_none(data, index + 1, axis=-1)

                variable_data = value[base_var][:, index]
            else:
                raise ValueError(f'Invalid input variable format : "{variable}". '
                                 f'Did you mean "{base_var}"?')
        # End of if statement (if '[' and ']' present in input variable)
        elif '[' in variable or ']' in variable:
            raise ValueError('Expect input variable to be "variable" or "variable[int]".'
                             'Perhaps you forgot a "[" or "]"?')
        else:
            raise ValueError(f"Variable '{variable}' not found in sample '{key}'. Available variable(s):"
                       f"{valid_var}")
    return variable_data

def stacked_histogram(data_dict, color_list, variable, xmin, xmax, num_bins, main_axes, marker, show_back_unc, residual_axes, x_label, label_fontsize, tick_labelsize):
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
    bullets = 1

    data_x = np.zeros(len(bin_centres))
    
    for (key, value), color in zip(data_dict.items(), color_list): # loop over samples
        if isinstance(value, dict):
            valid_var = list(value.keys())
        elif isinstance(value, ak.Array):
            valid_var = value.fields
        else:
            print(f'Key "{key}" : Unexpected type of dict value. Expect dict or Awkward Array.')
            raise TypeError
        # Validate the input variable
        variable_data = get_variable_data(variable, key, value, valid_var)
            
        if 'Data' in key:
            hist_data = Hist.new.Reg(num_bins, xmin, xmax, name=key).Double()
            # Fill values, replacing any None with nan first
            hist_data.fill(ak.to_numpy(ak.fill_none(variable_data, np.nan)))

            text.append(f'({bullets}) {key}: Sum (value = {sum(hist_data.view(flow=False)):.3e}),')
            text.append(f'Underflow = {hist_data.view(flow=True)[0]:.3e}, Overflow = {hist_data.view(flow=True)[-1]:.3e}\n')
            bullets += 1
        
            data_x = hist_data.view(flow=False)
            # print(f'data_x[:5] = {data_x[:5]}')
            data_x_errors = np.sqrt(data_x) # statistical error on the data
            hists.append(hist_data)
            
            # Plot the data points
            main_axes.errorbar(x=bin_centres, y=data_x, yerr=data_x_errors,
                                marker=marker, color=color, linestyle='none',
                                label=key) 
        elif 'Signal' in key:
            signal_x.append(ak.to_numpy(ak.fill_none(variable_data, np.nan))) # histogram the signal
            # print(f'signal_x[-1][:5] = {signal_x[-1][:5]}')
            if 'totalWeight' in valid_var:
                signal_weights.append(ak.to_numpy(value['totalWeight']))
            else:
                signal_weights.append(np.ones(len(variable_data)))
            signal_colors.append(color) # get the colour for the signal bar
            signal_labels.append(key)
            signal_input = True # MC signal present
        else:
            background_x.append(ak.to_numpy(ak.fill_none(variable_data, np.nan))) # append to the list of MC histogram entries
            # print(f'background_x[-1][:5] = {background_x[-1][:5]}')
            
            # append to the list of MC weights
            if 'totalWeight' in valid_var:
                background_weights.append(ak.to_numpy(value['totalWeight']))
            else:
                background_weights.append(np.ones(len(variable_data)))
            background_colors.append(color) # append to the list of MC bar colors
            background_labels.append(key) # append to the list of MC legend labels
            background_input = True # Background signal present
        
     
    if background_input: # Background signal present
        background_hists = []
        for back_data, back_weight, label in zip(background_x, background_weights, background_labels):
            back_hist = Hist.new.Reg(num_bins, xmin, xmax, name=label).Weight()
            back_hist.fill(back_data, weight=back_weight)
            text.append(f'({bullets}) {label}: Weighted Sum (value = {back_hist.sum().value:.3e}, '
                        f'variance = {back_hist.sum().variance:.3e}),')
            text.append(f'Underflow = {back_hist.view(flow=True)[0].value:.3e}, Overflow = {back_hist.view(flow=True)[-1].value:.3e}')
            bullets += 1
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
            main_axes.bar(x=bin_centres, height=counts, width=widths, bottom=bottom, color=color, label=label, align='center')
            bottom += counts

        if show_back_unc:
            # Plot the statistical uncertainty
            main_axes.bar(x=bin_centres, height=stat_err*2, width=widths, bottom=bottom-stat_err, color='none', hatch="////", label='Stat. Unc.', align='center')
        
    else:
        back_stacked_counts = np.zeros(len(bin_centres))

    # print(f'before signal_input, back_stacked_counts = {back_stacked_counts[:5]}')
    
    if signal_input: # MC signal present
        signal_hists = []
        for data, weight, label in zip(signal_x, signal_weights, signal_labels):
            signal_hist = Hist.new.Reg(num_bins, xmin, xmax, name=label).Weight()
            signal_hist.fill(data, weight=weight)
            # print(f'signal_hist.view(flow=False).value[:5] = {signal_hist.view(flow=False).value[:5]}')
            text.append(f'({bullets}) {label}: Weighted Sum (value = {signal_hist.sum().value:.3e}, '
                        f'Variance = {signal_hist.sum().variance:.3e}),')
            text.append(f'Underflow = {signal_hist.view(flow=True)[0].value:.3e}, '
                        f'Overflow = {signal_hist.view(flow=True)[-1].value:.3e}')
            bullets += 1
            signal_hists.append(signal_hist)
            hists.append(signal_hist)
            
        # Total count of signal data
        stacked_counts = sum(h.view(flow=False).value for h in signal_hists)
        # print(f'after in signal_input and after for loop, stacked_counts[:5] = {stacked_counts[:5]}')

        # Plot the bars
        bottom = back_stacked_counts.copy()
        # print(f'bottom[:5] = {bottom[:5]}')
        # print(f'back_stacked_counts[:5] = {back_stacked_counts[:5]}')
        
        for h, color, label in zip(signal_hists, signal_colors, signal_labels):
            counts = h.view(flow=False).value
            main_axes.bar(x=bin_centres, height=counts, width=widths, bottom=bottom, color=color,
                    label=label, align='center')
            bottom += counts
            # print(f'in the signal_hists for loop, back_stacked_counts = {back_stacked_counts[:5]}')
    else:
        stacked_counts = np.zeros(len(bin_centres))

    # print(f'after if signal_input, stacked_counts[:5] = {stacked_counts[:5]}')
    # print(f'back_stacked_counts[:5] = {back_stacked_counts[:5]}')
    mc_total = stacked_counts + back_stacked_counts
    # print(f'mc_total[:5] = {mc_total[:5]}')

    if residual_axes is not None:
        if not np.all(mc_total == 0) and not np.all(data_x == 0):
            # Calculate and plot residuals
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.divide(data_x, mc_total, out=np.full_like(data_x, np.nan), where=mc_total != 0)
                yerr = np.divide(np.abs(ratio * data_x_errors),
                                 data_x,
                                 out=np.zeros_like(data_x),  # set to 0 when undefined
                                 where=data_x != 0)
            # print(f'ratio[:5] = {ratio[:5]}')
            residual_axes.errorbar(bin_centres, ratio, yerr=yerr, fmt='ko')
            residual_axes.axhline(1, color='r', linestyle='--')
            residual_axes.set_xlabel(x_label, fontsize=label_fontsize,
                             x=1, horizontalalignment='right' )
            residual_axes.set_ylabel('Ratio (Data/MC)', fontsize=label_fontsize,
                             y=1, horizontalalignment='right')
            residual_axes.xaxis.set_minor_locator(AutoMinorLocator())
            residual_axes.yaxis.set_minor_locator(AutoMinorLocator())
            residual_axes.tick_params(which='both', direction='in', top=True, right=True)
            # set the axis tick parameters for the main axes
            residual_axes.tick_params(which='both', # ticks on both x and y axes
                                  direction='in', # Put ticks inside and outside the axes
                                  labelsize=tick_labelsize, # Label size
                                  top=True, # draw ticks on the top axis
                                  right=True) # draw ticks on right axis



    return bin_centres, hists, text


def validate_plotting_input(data_dict, color_list, num_bins, xmin, xmax, fig_size):
    if not isinstance(data_dict, dict):
        raise TypeError('data_dict must be a dict.')

    if not isinstance(color_list, (list, tuple)):
        raise TypeError('Input for colors must be a list.')

    # Check if sample keys and color list have the same length
    if len(data_dict) != len(color_list):
        raise ValueError("Input lists for sample keys and colors must have the same length.") 
        raise ValueError("Sample keys and color list must match in length. "
            f"Got {len(data_dict)} and {len(color_list)}")

    # Validate num_bins
    if not isinstance(num_bins, int):
        raise TypeError(f'num_bins must be an int. Got {num_bins}')
    if not num_bins > 1:
        raise ValueError(f'num_bins must be greater than one! Got "{num_bins}"')
        
    # Validate xmin and xmax
    if not isinstance(xmin, (int, float)):
        raise TypeError(f'x_min must be a number. Got "{x_min}"')
    if not isinstance(xmax, (int, float)):
        raise TypeError(f'x_max must be a number. Got "{x_max}"')
    if xmax <= xmin:
        raise ValueError(f'x_max must be greater than x_min. Got x_max = "{x_max}", x_min = "{x_min}"')

    # Validate fig_size
    if (not isinstance(fig_size, tuple)
        or not all(isinstance(value, (int, float)) for value in fig_size)):
        raise ValueError("fig_size must be a tuple of two numbers.")
    if len(fig_size) != 2 or not all(value > 0 for value in fig_size):
        raise ValueError("fig_size must be a tuple of two positive numbers.")


def plot_stacked_hist(data_dict, plot_variable, color_list,
                      num_bins, xmin, xmax, x_label,
        # Optional arg
        y_label=None, # Str for y axis label
        fit=None,
        fit_label='fit',
        fit_fmt='-r',
        logy=False, # Whether to set the y axis as log scale
        title=None, # Str or list of str for title
        marker='o', # Marker type
        title_fontsize=17, # Fontsize for title
        label_fontsize=17, # Fontsize for x and y axes
        legend_fontsize=17, # Fontsize for legend
        tick_labelsize=15, # Fontsize for x and y axes ticks
        text_fontsize=14, # Fontsize for text that shows histogram info
        fig_size=(12, 8), # Figure size
        show_text=False, # Bool - whether to show the text that displays histogram info
        show_back_unc=True, # Bool - whether to show the background uncertainty
        fig_name=None,
        residual_plot=False
       ):
    
    validate_plotting_input(data_dict, color_list, num_bins, xmin, xmax, fig_size)

    time_start = time.time()

    if residual_plot:
    # Create main plot and residual subplot
        fig, (main_axes, residual_axes) = plt.subplots(2, 1, figsize=fig_size, 
                                                       gridspec_kw={'height_ratios': [3, 1]}, 
                                                       sharex=True)
    else:
        fig, main_axes = plt.subplots(figsize=fig_size)
        residual_axes = None

    bin_centres, hists, text = stacked_histogram(data_dict, color_list, plot_variable, xmin, xmax, num_bins, main_axes, marker, show_back_unc, residual_axes, x_label, label_fontsize, tick_labelsize)

    if fit is not None:
        if len(fit) != len(bin_centres):
            raise ValueError('The array for fitted data must have the same length as the bin centres. Perhaps you used the wrong num_bins, xmin, or xmax?')
        main_axes.plot(bin_centres, fit, fit_fmt, label=fit_label)
        
    if show_text:
        if residual_plot:
            for i, line in enumerate(text):
                residual_axes.text(-0.05, -0.2 - i * 0.15, line, ha='left', va='top', transform=residual_axes.transAxes, fontsize=text_fontsize)
        else:
            for i, line in enumerate(text):
                main_axes.text(-0.05, -0.2 - i * 0.05, line, ha='left', va='top', transform=main_axes.transAxes, fontsize=text_fontsize)
    # separation of x axis minor ticks
    main_axes.xaxis.set_minor_locator(AutoMinorLocator()) 
    
    # set the axis tick parameters for the main axes
    main_axes.tick_params(which='both', # ticks on both x and y axes
                          direction='in', # Put ticks inside and outside the axes
                          labelsize=tick_labelsize, # Label size
                          top=True, # draw ticks on the top axis
                          right=True) # draw ticks on right axis

    if not residual_plot:
        # x-axis label
        main_axes.set_xlabel(x_label, fontsize=label_fontsize,
                             x=1, horizontalalignment='right' )

    # write y-axis label for main axes
    if not y_label:
        y_label = 'Events'
    main_axes.set_ylabel(y_label,
                         fontsize=label_fontsize,
                         y=1, horizontalalignment='right') 
    
    # draw the legend
    main_axes.legend(frameon=False, fontsize=legend_fontsize) # no box around the legend    
    
    main_axes.set_title(title, fontsize=title_fontsize)
    
    if logy:
        main_axes.set_yscale('log')
        if residual_plot:
            residual_axes.set_yscale('log')

    elapsed_time = time.time() - time_start 
    print("Elapsed time = " + str(round(elapsed_time, 1)) + "s") # Print the time elapsed

    # Show the plot
    plt.tight_layout()
    plt.show()

    if fig_name:
        fig.savefig(str(fig_name))

    return fig, hists
    

def histogram_2d(data, num_bins, min_max, label,
                 label_fontsize=12, tick_labelsize=10,
                 title_fontsize=13, title='', colorbar_label=''):
    # Validate variable
    if (not isinstance(data, (list, tuple)) or
        not all(isinstance(i, ak.Array) for i in data)
       ):
        raise TypeError("data must be a list or tuple of two awkward arrays.")
        
    # Validate the length of input
    if (
        len(data) != 2
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
        if j:
            label[i] = str(j)
        else:
            label[i] = variable[i]

    # Extract values for x and y from inputs
    data_x, data_y = data
    num_bins_x, num_bins_y = num_bins
    (xmin, xmax), (ymin, ymax) = min_max
    label_x, label_y = label

    # Histogram
    h = Hist(
        hist.axis.Regular(num_bins_x, xmin, xmax, name=label_x, label=label_x, flow=False),
        hist.axis.Regular(num_bins_y, ymin, ymax, name=label_y, label=label_y, flow=False)
    )
    h.fill(ak.to_numpy(data_x), ak.to_numpy(data_y))

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
    cbar.set_label(colorbar_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=tick_labelsize)
    plt.show()
    return fig, h

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Below are functions to plot made to plot variables separately

# Helper functions
def validate_xlabel(x_label_list, plot_variables):
    variable_count = len(plot_variables)
    x_label = []
    if isinstance(x_label_list, str): # If input is a string
        x_label = [x_label_list] * variable_count
    elif isinstance(x_label_list, (list, tuple)):
        if len(x_label_list) == 0: # If input is an empty list
            for variable in plot_variables:
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
    if y_label_list is None:
        y_label_list = 'Events'
    y_label = []
    if isinstance(y_label_list, str): # If input is a string
        y_label = [y_label_list] * variable_count
    elif isinstance(y_label_list, (list, tuple)):
        if len(y_label_list) == 0: # If input is empty
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
    if title_list is None:
        title_list = ''
    title = []
    if isinstance(title_list, str): # If input is a string
        title = [title_list] * variable_count
    elif isinstance(title_list, (list, tuple)):
        if len(title_list) == 0: # If input is empty
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


# This function aims to plot one or more variables separately
def plot_histograms(
        data_dict,
        plot_variables,
        color_list,
        xmin_xmax_list,
        num_bins_list,
        x_label_list,
        # Optional arguments start from here
        y_label_list=None, # Str or list of str for y axis label
        logy=False, # Whether to set the y axis as log scale
        title_list=None, # Str or list of str for title
        marker='o', # Marker type
        title_fontsize=17, # Fontsize for title
        label_fontsize=17, # Fontsize for x and y axes
        legend_fontsize=17, # Fontsize for legend
        tick_labelsize=15, # Fontsize for x and y axes ticks
        text_fontsize=14, # Fontsize for text that shows histogram info
        fig_size=(12, 8), # Figure size
        show_text=False, # Bool - whether to show the text that displays histogram info
        show_back_unc=True, # Bool - whether to show the background uncertainty
        residual_plot=False
    ):
    if not isinstance(data_dict, dict):
        raise TypeError('data_dict must be a dict.')

    if isinstance(color_list, str):
        raise TypeError('Input for colors must be a list.')

    # Check if all input lists are the same length
    if len(data_dict) != len(color_list):
        raise ValueError("Input lists for sample keys and colors must have the same length.") 
        raise ValueError("Sample keys and color list must match in length. "
            f"Got {len(data_dict)} and {len(color_list)}")


    # Validate fig_size
    if (not isinstance(fig_size, tuple)
        or not all(isinstance(value, (int, float)) for value in fig_size)):
        raise ValueError("fig_size must be a tuple of two numbers.")
    if len(fig_size) != 2 or not all(value > 0 for value in fig_size):
        raise ValueError("fig_size must be a tuple of two positive numbers.")

    time_start = time.time()

    # Validate plot_variables
    if isinstance(plot_variables, (list, tuple)):
        variable_count = len(plot_variables)
    else:
        variable_count = 1 # It is a str
        plot_variables = [plot_variables] * variable_count

    # Validate x label input
    # If x_label_list is one string, make it into an array corresponding to the length of other input list
    x_label_list = validate_xlabel(x_label_list, plot_variables)

    # Validate ylabel input
    y_label_list = validate_ylabel(y_label_list, variable_count)
    
    # Validate title
    title_list = validate_title(title_list, variable_count)
    
    # Validate number of bins input
    num_bins_list = validate_num_bins(num_bins_list, variable_count)
        
    # Validate xmin xmax input                
    xmin_xmax_list = validate_xmin_xmax(xmin_xmax_list, variable_count)

    fig_list = []
    hists_list = []
                
    for (variable, x_label, y_label,
         xmin_xmax, num_bins, title) in zip(plot_variables, x_label_list,
                                                y_label_list, xmin_xmax_list,
                                                num_bins_list, title_list):
        
        if residual_plot:
        # Create main plot and residual subplot
            fig, (main_axes, residual_axes) = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        else:
            fig, main_axes = plt.subplots(figsize=fig_size)
            residual_axes = None
        
        xmin, xmax = xmin_xmax
        _, hists, text = stacked_histogram(data_dict, color_list, variable, xmin, xmax, num_bins, main_axes, marker, show_back_unc, residual_axes, x_label, label_fontsize, tick_labelsize)

        if show_text:
            if residual_plot:
                for i, line in enumerate(text):
                    residual_axes.text(-0.05, -0.2 - i * 0.15, line, ha='left', va='top', transform=residual_axes.transAxes, fontsize=text_fontsize)
            else:
                for i, line in enumerate(text):
                    main_axes.text(-0.05, -0.2 - i * 0.05, line, ha='left', va='top', transform=main_axes.transAxes, fontsize=text_fontsize)
        
            
        # separation of x axis minor ticks
        main_axes.xaxis.set_minor_locator(AutoMinorLocator()) 
        
        # set the axis tick parameters for the main axes
        main_axes.tick_params(which='both', # ticks on both x and y axes
                              direction='in', # Put ticks inside and outside the axes
                              labelsize=tick_labelsize, # Label size
                              top=True, # draw ticks on the top axis
                              right=True) # draw ticks on right axis
    
        if not residual_plot:
            # x-axis label
            main_axes.set_xlabel(x_label, fontsize=label_fontsize,
                                 x=1, horizontalalignment='right' )
                
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

        
        hists_list.append(hists)
        fig_list.append(fig)

    return fig_list, hists_list
    