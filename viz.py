#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:25:07 2019

@author: makinea1
"""

import numpy as np
from mayavi import mlab
import os

import matplotlib
cmap20 = matplotlib.cm.get_cmap('tab20c')
colors = cmap20(np.arange(20)/19)[:,:3]

pickup_R = 0.014

path = '.'
transforms = dict(np.load(os.path.join(path, 'array120_trans_newnames.npz')))

def pickup_path():
    c = np.zeros((4, 5))
    c[:2, 0] = [-1, -1]
    c[:2, 1] = [1, -1]
    c[:2, 2] = [1, 1]
    c[:2, 3] = [-1, 1]
    c[:2, 4] = [-1, -1]
    c *= pickup_R

    c[3] = 1
    return c

def rotationToVtk(R):
    '''
    Concert a rotation matrix into the Mayavi/Vtk rotation paramaters (pitch, roll, yaw)
    '''
    def euler_from_matrix(matrix):
        """Return Euler angles (syxz) from rotation matrix for specified axis sequence.
        :Author:
          `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

        full library with coplete set of euler triplets (combinations of  s/r x-y-z) at
            http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

        Note that many Euler angle triplets can describe one matrix.
        """
        # epsilon for testing whether a number is close to zero
        _EPS = np.finfo(float).eps * 5.0

        # axis sequences for Euler angles
        _NEXT_AXIS = [1, 2, 0, 1]
        firstaxis, parity, repetition, frame = (1, 1, 0, 0) # ''

        i = firstaxis
        j = _NEXT_AXIS[i+parity]
        k = _NEXT_AXIS[i-parity+1]

        M = np.array(matrix, dtype='float', copy=False)[:3, :3]
        if repetition:
            sy = np.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
            if sy > _EPS:
                ax = np.arctan2( M[i, j],  M[i, k])
                ay = np.arctan2( sy,       M[i, i])
                az = np.arctan2( M[j, i], -M[k, i])
            else:
                ax = np.arctan2(-M[j, k],  M[j, j])
                ay = np.arctan2( sy,       M[i, i])
                az = 0.0
        else:
            cy = np.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
            if cy > _EPS:
                ax = np.arctan2( M[k, j],  M[k, k])
                ay = np.arctan2(-M[k, i],  cy)
                az = np.arctan2( M[j, i],  M[i, i])
            else:
                ax = np.arctan2(-M[j, k],  M[j, j])
                ay = np.arctan2(-M[k, i],  cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        return ax, ay, az
    r_yxz = np.array(euler_from_matrix(R))*180/np.pi
    r_xyz = r_yxz[[1, 0, 2]]
    return r_xyz

def rotationmat(angle):
    """ Rotation matrix from axis (dir of angle) and angle (mag of angle)
    """
    m = np.sqrt(angle[0]**2+angle[1]**2+angle[2]**2)
    if m < 1e-12:
        return np.eye(3)
    a, b, c = angle/m
    cross = np.array([[0, -c, b], [c, 0, -a], [- b, a, 0]])
    return np.eye(3) + np.sin(m)*cross + (1 - np.cos(m))*cross@cross


def plot_sensor_data(channels, data, cmap = 'viridis',
                     bads = [], vmin=None, vmax=None, opacity=1.0,
                     pcolor=(0.5, 0.5, 0.5), group_coloring=False,
                     plot_names=False):
    """ Plot sensor data in the MEG helmet as colors on the pickup coils

        Parameters:

            channels : array of channel names
            data : array of data (in the same order of channels)
            cmap : name of the colormap, passed to mayavi
            bads : list of bad channel names (pickup to be not colored)
            vmin : colormap minimum, passed to mayavi
            vmax ; colormap maximum, passed to mayavi
            opacity : scalar (0,1) opacity of the colors
            pcolor : color of the "pickup coil wire"
            group_coloring: False does nothing, True ignores pcolor and
                            colors the wires with different color for each
                            channel group
            plot_names: toggle channel name plotting
        Returns:

            mayvavi Surface object representing the data coloring
    """
    pickup = pickup_path()
    p = np.array([0,0,0,1]) # position vertor, origin

    points = np.array([(t@p)[:3] for t in transforms.values()])
    pickups = np.array([(t@pickup)[:3] for t in transforms.values()])

    names = list(transforms.keys())
    data_dict=dict(zip(channels,data))

    tris_pickup = np.array([[0,1,2],[0,2,3]])
    tris = []
    verts = []
    scalars = []
    jj = 0
    for i,p in enumerate(pickups):
        name = names[i]
        if group_coloring:
            color = tuple(colors[int(name[3:5]) % 20])
        else:
            color = pcolor
        mlab.plot3d(*p, color=color, tube_radius=0.0005)
        #if name in channels and name not in bads:
        if name in channels:
            verts.extend(p[:,:-1].T)
            tris.append(tris_pickup[0] + jj)
            tris.append(tris_pickup[1] + jj)
            for kk in range(4):
                scalars.append(data_dict[name])
            jj += 4
#            s = mlab.triangular_mesh(*p, np.array([[0,1,2],[0,2,3]]),
#                                 color=cmap(data_dict[name])[:3], opacity=opacity)
#            s.actor.property.lighting = False
    verts = np.array(verts)
    scalars = np.array(scalars)
    tris = np.array(tris)
    s = mlab.triangular_mesh(*verts.T, tris, scalars=scalars,
                         opacity=opacity, colormap=cmap,
                         vmin=vmin, vmax=vmax)

    if plot_names:
        points_names = points
        Rtext = (rotationmat(np.pi*np.array([1,0,0]))
                @ rotationmat(np.pi/2*np.array([0,0,1])))
        Rtext2 = rotationmat(np.pi/2*np.array([0,0,1]))
        for p,n,t in zip(points_names,names,transforms.values()):

            if n in bads:
                n = "BAD"
                color = (1, 0, 0)
            elif "MEG" in n:
                n = n[3:]
                color = (0,0,0)
            angles = rotationToVtk(t[:3,:3] @ Rtext)
            angles2 = rotationToVtk(t[:3,:3] @ Rtext2)
    #            print(angles)
            pn = p + pickup_R*(0.9*t[:3,0] + t[:3,1])*0.9  -t[:3,2]*0.001
            pn2 = p + pickup_R*(0.9*t[:3,0] - t[:3,1])*0.9  +t[:3,2]*0.001
    #            mlab.plot3d([p[0], pn[0]],[p[1], pn[1]],[p[2], pn[2]],
    #                        color=(0,0,0), tube_radius=None)
            t1 = mlab.text3d(*pn, n, color=color, scale=0.006,
                        orient_to_camera=False, orientation=angles)
            t1.actor.property.backface_culling = True
            t2 = mlab.text3d(*pn2, n, color=color, scale=0.006,
                        orient_to_camera=False, orientation=angles2)
            t2.actor.property.backface_culling = True

    return s

if __name__ == '__main__':
    bads = ['MEG2131', 'MEG2511']
    data = np.load('/m/nbe/project/megmri/megin_phantom/2019-09-20_pos2/field_dipole1.npz')
    s = plot_sensor_data(data['ch_names'], data['field'].real, bads = bads, opacity=0.8)
