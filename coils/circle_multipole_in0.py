# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:48:34 2013

@author: k7hoven
"""

from sympy import symbols, Ynm, pi


def inner_multipole_circle(l,m,radius):
    theta, phi = symbols("theta phi")
    Y = Ynm(l,m,theta,phi) # Ylm was changed to Ynm in sympy. Hope same definition?
    dY_dtheta = Y.conjugate().diff(theta).subs(theta,0)
    prefactor = radius**(-l)/((2*l+1)*l)
    integral = dY_dtheta.integrate((phi,0,2*pi))
    return prefactor*integral
    
if __name__ = '__main__':
    m = inner_multipole_circle(3,0,1)
    print(m)

