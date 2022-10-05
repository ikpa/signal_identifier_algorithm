# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 01:51:28 2013

Tools that help in vectorizing numerical stuff for speed.

@author: Koos Zevenhoven
"""

def sum_tail_axes(arr, n_axes):
    """Sum over the last n_axes axes of the array arr."""
    arr = arr.reshape(arr.shape[:-n_axes] + (-1,))
    return arr.sum(axis = arr.ndim - 1)

def sum_keep_axis(arr, axis = 0):
    """Sum over an axis without removing the axis from the result."""
    shape = list(arr.shape)
    arr = arr.sum(axis=axis)
    shape[axis] = 1         
    return arr.reshape(shape)

def insert_dummy_axis(arr, axis):
    """Return arr but with an axis added at position axis (None for end)."""
    shape = list(arr.shape)
    if axis == None:
        shape.append(1)
    else:
        shape.insert(axis,1)
    return arr.reshape(shape)
    
def left_align_axes(arr, ref_shape):
    """Return arr with axes appended up to number of axes in ref_shape."""
    if type(ref_shape) != tuple:
        ref_shape = ref_shape.shape
    new_shape = arr.shape + (len(ref_shape) - len(arr.shape))*(1,)
    return arr.reshape(new_shape)
    