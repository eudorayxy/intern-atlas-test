import hist
from hist import Hist 
import awkward as ak
import numpy as np

def get_histogram(variable_data, num_bins, xmin, xmax, hist_name, weight=None):
    # Replace any None with np.nan, and then convert to numpy
    variable_data = ak.to_numpy(ak.fill_none(variable_data, np.nan))
    if weight is not None:
        weight = ak.to_numpy(ak.fill_none(weight, 0.0))
        h = Hist.new.Reg(num_bins, xmin, xmax, name=hist_name).Weight()
        h.fill(variable_data, weight=weight)
        view = h.view(flow=False)
        value = view.value
        variance = view.variance
    else:
        h = Hist.new.Reg(num_bins, xmin, xmax, name=hist_name).Double()
        h.fill(variable_data)
        value = h.view(flow=False)
        variance = None
        
    bin_centres = h.axes[0].centers
    
    return value, variance, bin_centres