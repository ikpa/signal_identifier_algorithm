# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:52:00 2013

A module for handling thin conductive magnetic shields.

This is based on earlier Matlab code by Koos Zevenhoven 2010/2011.

@author: Koos Zevenhoven
"""
from __future__ import unicode_literals, division
import numpy as np
import pylab as pl

from .surfacecurrent import CoilSystemList, EddyPlate

class UNIT:
    INCH = 0.0254
    FOOT = 12.0 * INCH

class SIGMA:
    Al6061 = 1./3.7e-8

class SplitBoxRoom(CoilSystemList):
    """Describes a box-shaped thin shield with disconnected plates."""

    def __init__(self, 
                 side_divisions = [((8,), (2,4,2))] * 4,
                 floor_ceiling_divisions = [((4,4), (4,4))] * 2, 
                 unit_in_m = UNIT.FOOT,
                 conductivity = SIGMA.Al6061,
                 thickness = UNIT.INCH / 16,
                 detail_scale = UNIT.FOOT,
                 label = None):
        """Room with disconnected plates (default = new Berkeley room).
        
        The side_divisions are given as first the walls parallel to the
        x axis and then those parallel to the y axis. The first number in
        each pair is the sizes of "rows" and the second the sizes of
        "columns".
        
        Look at the floor and ceiling from above (z-direction), row
        sizes are measured in the x direction and column sizes in y.
        
        """
        from itertools import chain
        from . import rotmatrix
        pi = np.pi
        
        self.conductivity = conductivity
        self.thickness = thickness

        
        height = [sum(div[0]) for div in side_divisions]
        if any(h != height[0] for h in height):
            raise Exception("Room walls not compatible in vertical size.")
        height = height[0]
        
        x_width = ([sum(div[1]) for div in side_divisions[0:2]] + 
                   [sum(div[1]) for div in floor_ceiling_divisions])
        y_width = ([sum(div[1]) for div in side_divisions[2:]] +
                   [sum(div[0]) for div in floor_ceiling_divisions])
        
        if any(w != x_width[0] for w in x_width):
            raise Exception("Room sides not compatible in x size.")
        if any(w != y_width[0] for w in y_width):
            raise Exception("Room sides not compatible in y size.")
            
        widths = [sum(div[1]) for div in side_divisions]
        
        side_rotation_angles = (0, pi, pi/2, 3*pi/2)
        distances_from_origin = (widths[i % 4]/2 for i in range(1,5))
        
        def make_plates(divs, angles, dists, width_list, height, labels,
                        walls = True):
            for div,angle,dist,width,label in zip(divs,
                                                  angles,
                                                  dists,
                                                  width_list,
                                                  labels):
                if walls:
                    rot = rotmatrix.around_z(angle)
                else: # floor or ceiling
                    # rotate to ceiling position, fix floor later
                    rot = rotmatrix.around_x(-pi/2)
                    
                eh = unit_in_m * rot.dot(np.array([1,0,0]))
                ev = unit_in_m * rot.dot(np.array([0,0,1]))
                wall_center = unit_in_m * dist * rot.dot(np.array([0,-1,0]))
               
                if not walls:
                    wall_center *= angle #not angle, picks floor/ceiling                    
                    
                uleft = wall_center - width/2*eh + height/2*ev          
                
                uleft_hshifts = np.cumsum(div[1]) - np.array(div[1])/2           
                uleft_vshifts = np.cumsum(div[0]) - np.array(div[0])/2
                
                wall_plates = []
                for vshift, h in zip(list(uleft_vshifts), 
                                     list(np.array(div[0]))):
                    for hshift, w in zip(list(uleft_hshifts),
                                         list(np.array(div[1]))):
                        center = uleft + eh * hshift - ev * vshift
                        h_vec = eh * w
                        v_vec = ev * h
                        h_order = int(round(w / (detail_scale/unit_in_m)))
                        v_order = int(round(h / (detail_scale/unit_in_m)))
                        wall_plates.append(
                            EddyPlate(center, (h_vec, v_vec), 
                                      (h_order, v_order),
                                      label = "WxH={:0}x{:0}".format(w,h)))
        
                yield CoilSystemList(wall_plates, label = label)
                        
        wall_plates = make_plates(side_divisions, 
                                  side_rotation_angles, 
                                  distances_from_origin, 
                                  widths,
                                  height,
                                  map(lambda x: "Wall_%d" % x, range(1,5)))

        horiz_plates = make_plates(floor_ceiling_divisions,
                                   (-1,1), #no angles -- means floor, Roof
                                   (height/2,)*2,
                                   (x_width[0],)*2,
                                   y_width[0],
                                   ("Floor", "Roof"), 
                                   walls = False)
       
        sides = chain(wall_plates, horiz_plates)

        if label == None:
            label = "SplitBoxRoom"

        CoilSystemList.__init__(self, sides, label = label)

dipole_c = None
field_c = None

def transient_amplitudes(system, t, waveform, t_calc, 
                         dipole_p, dipole, field_p):
    global dipole_c, field_c
    
    if dipole_c == None:
        dipole_c = system.coupling_vector_dipole(dipole_p,dipole)
        field_c = system.generated_fields(field_p)
    
    mode_input = - (system.V.T.dot(dipole_c)) / system.lambdas
    mode_output = system.V.T.dot(field_c.T)
    
    dP = np.diff(np.concatenate((waveform, np.array([0]))))
    
    feed = mode_input.reshape(1,-1) * dP.reshape(-1,1)
    resp = np.zeros((np.size(t_calc), np.size(system.tau)))
    
    for i in range(len(t_calc)):
        tt = t-t_calc[i]
        tt_per_tau = tt.reshape(-1,1) * (1./system.tau).reshape(1,-1)
        
        resp[i,:] = np.sum(np.exp(tt_per_tau[tt<=0,:])*feed[tt<=0,:], axis=0)
    
    return resp.dot(mode_output)
    
def diagonalize(system):
    #TODO make this part of the great coil-system/lin-system class hierarchy
    system.lambdas, system.V = np.linalg.eigh(system.M)
    system.tau = system.lambdas * system.thickness * system.conductivity

def transient_test():
    newcube = SplitBoxRoom(detail_scale = 2*UNIT.FOOT)
    diagonalize(newcube)
    pl.ion()
    pl.imshow(np.log10(np.abs(newcube.M)), interpolation = 'nearest')
    pl.colorbar()
    
    n = 240
    r = 0.19
    I = 200 
    dipole_p = np.array([0,0,0])
    dipole = np.array([0,0,1]) * n * I * np.pi * r**2
    
    field_p = np.array([0,0,0])
    ramp_time = 0.010
    
    t = np.linspace(-ramp_time, 0, 10*40)
    Pwave = np.sin(-t/ramp_time * np.pi/2)
    t_calc = np.linspace(-ramp_time, 0.050, 60*10)
    
    field = transient_amplitudes(newcube, t, Pwave, t_calc, 
                                 dipole_p, dipole, field_p)
    
    pl.figure(4)
    pl.plot(t_calc, field)


if __name__ == "__main__":
    transient_test()
    
#    
#    newcube = SplitBoxRoom(detail_scale = 2 * UNIT.FOOT)
#    M = newcube.mutual_inductances(newcube)
#    M[M==0] = 1
#
#    pl.figure()
#    newcube.plot_geometry()
#    
