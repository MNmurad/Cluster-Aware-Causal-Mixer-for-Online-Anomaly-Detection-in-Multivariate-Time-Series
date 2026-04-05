
from skimage.measure import block_reduce # for downsample
import numpy as np

def downsample_data(list_arr, downsample, reduce_method=np.median, all_should_be_anomaly=True):
    # list_arr contains a list of array with the first dimension as downsample target
    # assume if there are only 0, 1 in an array, then it is the labels array
    new_list_arr = []
    for arr in list_arr:
        if len(arr.shape) == 1 and (arr == 0).sum() + (arr == 1).sum() == np.prod(arr.shape):
            reduce_func = np.min if all_should_be_anomaly else np.max
            block_sizes = (downsample, )
        else:
            reduce_func = reduce_method
            block_sizes = (downsample, 1)
        new_len = len(arr) // downsample * downsample
        arr = block_reduce(arr[:new_len], block_sizes, reduce_func) 
        new_list_arr.append(arr)

    return new_list_arr