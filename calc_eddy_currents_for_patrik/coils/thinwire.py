# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 16:01:49 2013

A module for handling coils wound of thin wire.

This is based on earlier Matlab code by Koos Zevenhoven 2008-2009.

This module is to be merged into a more general field calculation package.

@author: Koos Zevenhoven
"""
import numpy as np

def dipole_flux_loop_origin(path):
    """Compute the Biot--Savart integral for a path w.r.t. the origin.
    
    Arguments:
        path: Array of shape (3, n) with 3-vectors of positions.
            For a closed loop the first vector should equal the last one.
    """
    path = np.asarray(path)
    path2 = path[:,1:]
    path1 = path[:,:-1]
    Rsq = np.sum(path*path, 0);
    R = np.sqrt(Rsq);
    R2sq = Rsq[1:];
    R1sq = Rsq[:-1];
    R2 = R[1:]
    R1 = R[:-1]
    
    cross12 = np.cross(path1, path2, axis = 0)
    dot12 = np.sum(path1 * path2, axis = 0)
    
    tmp = R1+R2
    tmp = tmp / (R1sq * R2sq + R1 * R2 * dot12)
     
    return np.sum(cross12*tmp, axis=1) # should be times 1e-7 = mu0/4pi

def biot_savart_edge(r1, r2, points):
    """Biot--Savart integral for a line from r1 to r2, at given points.
    
    Arguments:
        r1: line starting point (3-vector)
        r2: line end point (3-vector)
        points: array of field points (first axis has dimension 3)
    """
    
    from .vectorizetools import left_align_axes
    
    r1 = left_align_axes(r1, points) - points
    r2 = left_align_axes(r2, points) - points
    
    R1sq = (r1**2).sum(axis = 0)
    R2sq = (r2**2).sum(axis = 0)
    R1 = np.sqrt(R1sq)
    R2 = np.sqrt(R2sq)
    
    cross12 = np.cross(r1, r2, axis = 0)
    dot12 = np.sum(r1 * r2, axis = 0)
    
    tmp = (R1 + R2) / (R1sq * R2sq + R1 * R2 * dot12)
    
    return cross12 * tmp
    
def biot_savart_polyline(vertices, field_points):
    """Calculate the Biot-Savart integral over a polyline.
    
    The integral is not multiplied by mu0/4pi.
    
    Arguments:
        vertices: vertices of the polyline path to integrate over, has
            shape (3, n), where n is the number of vertices. For a closed
            loop, vertices 0 and n-1 should be equal.
        field_points: The 'field points' at which to calculate the integral.
            The first axis (0) should be of dimension 3.
    """
    
    vertices = np.asarray(vertices)
    field_points = np.asarray(field_points)
    
    result = np.zeros(field_points.shape)
    
    for i in range(vertices.shape[1] - 1 ):
        result += biot_savart_edge(vertices[:, i], vertices[:, i + 1], 
                                   field_points)
        
    return result
                
        
    
    
    
    