# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:17:20 2013

Rotation matrices etc.

@author: Koos Zevenhoven
"""
from __future__ import unicode_literals, division

from numpy import array as arr, pi, sin



def around_x(angle):
    halfpi = 0.5*pi
    return sin(arr([[halfpi,    0,              0               ],
                    [0,         halfpi - angle, -angle          ],
                    [0,         angle,          halfpi - angle  ]]))

def around_y(angle):
    halfpi = 0.5*pi
    return sin(arr([[halfpi - angle,    0,         angle         ],
                    [0,                 halfpi,    0             ],
                    [-angle,            0,         halfpi - angle]]))

def around_z(angle):
    halfpi = 0.5*pi
    return sin(arr([[halfpi - angle,    -angle,         0       ],
                    [angle,             halfpi - angle, 0       ],
                    [0,                 0,              halfpi  ]]))
                    