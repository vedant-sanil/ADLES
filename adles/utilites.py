import numpy as np
import bisect

def binary_search(time, time_list):
    
    return time_list[bisect.bisect_left(time_list, time)]