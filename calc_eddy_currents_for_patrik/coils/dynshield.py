# -*- coding: utf-8 -*-
"""
Created on Sat Aug 03 17:42:20 2013

A module for simulating dynamic shielding and optimizing waveforms.

Still crap.

On 24 March 2014, I moved this into an ipython notebook. When more mature,
this should be moved back to a module.

@author: Koos Zevenhoven
"""
from __future__ import unicode_literals, division
from .eddycurrents0 import diagonalize, transient_amplitudes, SplitBoxRoom, UNIT
from .coils import ScalarCoilPlate
import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs

class Memoize:
	def __init__ (self, f):
		self.f = f
		self.mem = {}
	def __call__ (self, *args, **kwargs):
		if (args, str(kwargs)) in self.mem:
			return self.mem[args, str(kwargs)]
		else:
			tmp = self.f(*args, **kwargs)
			self.mem[args, str(kwargs)] = tmp
			return tmp

scoil_c = None
field_c = None

def shield_transient(system, coil, t, waveform, t_calc, field_p):
    global scoil_c, field_c
    
    if scoil_c == None:
        scoil_c = coil.mutual_inductances(system)
        field_c = system.generated_fields(field_p)
    
    mode_input = - (system.V.T.dot(scoil_c.flatten())) / system.lambdas
    mode_output = system.V.T.dot(field_c.T)
    
    dP = np.diff(np.concatenate((waveform, np.array([0]))))
    
    feed = mode_input.reshape(1,-1) * dP.reshape(-1,1)
    resp = np.zeros((np.size(t_calc), np.size(system.tau)))
    
    for i in range(len(t_calc)):
        tt = t-t_calc[i]
        tt_per_tau = tt.reshape(-1,1) * (1./system.tau).reshape(1,-1)
        
        resp[i,:] = np.sum(np.exp(tt_per_tau[tt<=0,:])*feed[tt<=0,:], axis=0)
    
    return resp.dot(mode_output)

def additional_mode(tau, t, wave, io_product):
    d_wave = np.diff(np.concatenate((wave, np.array([0]))))
    exponential = np.exp(t/tau)
    return - io_product * np.sum(exponential * d_wave)
    

result = None
current = None
opt_thread = None
add_total = None
def dynshield_berkeley():
    global t_dyn, waveform, ramp_time, t_calc
    newcube = SplitBoxRoom(detail_scale = UNIT.FOOT)
    diagonalize(newcube)
    
    s_side = 1.85
    s_turns = 30
    s_res = 20
    s_R = 30/40*1.2
    s_L = (30/40)**2*10e-3
    s_center = -4*UNIT.FOOT + 1.125
    s_coil = ScalarCoilPlate(s_center,
                             (np.array([s_side,0,0]),np.array([0,s_side,0])),
                             np.ones((1,s_res,s_res))*s_turns,
                             label = "shielding_coil")
    n = 240
    r = 0.19
    I = 200
    dipole_p = np.array([0,0,-3.0 * 0.0254])
    dipole = np.array([0,0,1]) * n * np.pi * r**2
    
    field_p = np.array([0,0,0])
    ramp_time = 0.0116
    
    t = np.linspace(-ramp_time, 0, int(ramp_time/0.1e-3))
    Pwave = np.sin(-t/ramp_time * np.pi/2)
    Pwave[t<-ramp_time] = 1
    Pwave *= I
    
    waveform_start = -0.040
    t_dyn = np.linspace(waveform_start, -0.000, int(-waveform_start/.1e-3))
    waveform = np.zeros(t_dyn.shape)
    
    deltat = (t_dyn[1]-t_dyn[0])
    
    instants = np.arange(1,10)*1e-3
    trans = transient_amplitudes(newcube, t, Pwave,
                                 instants, 
                                 dipole_p, dipole, field_p)[:,2]

    io_prod_bp = 0.812
    io_prod_canc = 3.51
    add_weight = 1e-8
    addmode = additional_mode(23e-3, t, Pwave, io_prod_bp)
    
    pis = [1,2,3,4,5] 
    pows = [1,2,1,2,1]
    tautaus = np.array(arange(2,25))*1e-3
    #tautaus = np.array([1.2, 5.5, 9, 20])*1e-3
    #tautaus = np.array([1,2,4,8,12,18, 23])*1e-3    
    n_basis = len(tautaus)
    basis = np.zeros((t_dyn.size, n_basis))

    for i in range(n_basis):
        funk = lambda x, i: np.exp(-x**4/tautaus[i]**4)*(-x)
        x = t_dyn
        basis[:,i] = funk(x,i)/np.max(funk(x,i))#p.sin(x)**2
    
    plt.figure(1)
    plt.plot(x, basis)
    
    def power_and_voltage(waveform):
        energy = deltat * np.sum(waveform**2)
        voltage = waveform[1:] * s_R + s_L * np.diff(waveform)/deltat
        return energy, voltage

        
    def waveform_from_x(x):
        #waveform[1:-1] = x
        waveform[:] = basis.dot(x)
        return waveform
    
        
    def penalty(x):
        global add_total
        waveform[:] = waveform_from_x(x)
        canc = shield_transient(newcube, s_coil, t_dyn, waveform,
                                instants, field_p)
        canc = canc[:,2]
        
        add_total = addmode + \
            additional_mode(23e-3, t_dyn, waveform, io_prod_canc)
        value = (np.sqrt(np.mean((trans+canc)**2) + \
                         (add_weight * add_total)**2)*1e6)**2        
        #print canc
        #print trans
        energy, voltage = power_and_voltage(waveform)
        
        #d2 = np.diff(waveform, 2)
        
        #value += 1e-22 * ((np.diff(waveform)/deltat)**4).mean()
        #value += 1e-4 * (waveform**2).sum() * deltat
        overvoltage = np.abs(voltage) - 120
        overvoltage[overvoltage < 0] = 0
        
        v_pow = 2
        value += 1e-4*np.mean(overvoltage**v_pow)**(1/v_pow)
        #value += 1e-5 * np.sqrt(np.mean(d2**2))
        
        return value
    
    def callback(x):
        global current
        callback.count += 1
        waveform[:] = waveform_from_x(x)
        if callback.count % 10 == 0:
            print("iteration ", callback.count)
            #print "max_field ", np.abs(trans + canc).max()
            #print "value ", value
        print("additional ", add_total)
        current = waveform
            

    callback.count = 0
    
    from threading import Thread
    
    def find_waveform():
        global result

        #x0 = waveform[1:-1]
        x0 = np.zeros([n_basis])
#        result = minimize(penalty, x0, method = "BFGS", jac = False,
#                      options={ 'disp' : True, 'maxiter' : 300 },
#                      callback=callback)
        result = fmin_bfgs(penalty, x0, fprime=None, args=(),
                           gtol = 1e-6, maxiter = 300, disp=1, callback=callback)
        #waveform[:] = waveform_from_x(result.x)
        waveform[:] = waveform_from_x(result)
        
    global opt_thread
    opt_thread = Thread(target=find_waveform)
    opt_thread.start()
    wf = np.zeros(t_dyn.shape)
    
    globals().update(locals())

plt.ion()

def plot_current():
    global t_dyn, t_calc
    plt.figure(5)
    plt.plot(t_dyn, current)
        
    t_calc = np.linspace(-ramp_time, 0.050, 60*10)     
    field = transient_amplitudes(newcube, t, Pwave, t_calc, 
                                 dipole_p, dipole, field_p)
    field += shield_transient(newcube, s_coil, 
                              t_dyn, waveform, 
                              t_calc, field_p)
    
    instant = 1e-3
    canc = shield_transient(newcube, s_coil, 
                            t_dyn, waveform,
                            np.array([instant]), 
                            field_p)[0,2]*1e6
    trans = transient_amplitudes(newcube, t, Pwave, np.array([instant]), 
                                 dipole_p, dipole, field_p)[0,2]*1e6              
    
    print("Transient at origin (t = %f ms): %f uT" % (instant * 1000, canc + trans))
    plt.figure(4)
    plt.plot(t_calc, field)
    


def plot_results():
    global t_dyn, t_calc
    opt_thread.join()
    plt.figure(5)
    plt.plot(t_dyn, waveform)
    #pl.plot(t, Pwave)
    
    t_calc = np.linspace(-ramp_time, 0.050, 60*10)     
    field = transient_amplitudes(newcube, t, Pwave, t_calc, 
                                 dipole_p, dipole, field_p)
    field += shield_transient(newcube, s_coil, 
                              t_dyn, waveform, 
                              t_calc, field_p)
    
    instant = 1e-3
    canc = shield_transient(newcube, s_coil, 
                            t_dyn, waveform,
                            np.array([instant]), field_p)[0,2] * 1e6
    trans = transient_amplitudes(newcube, t, Pwave, np.array([instant]), 
                                 dipole_p, dipole, field_p)[0,2] * 1e6              
    
    print("Transient at origin (t = %f ms): %f uT" % (instant * 1000, canc + trans))
    plt.figure(4)
    plt.plot(t_calc, field)
    

if __name__ == "__main__":
    dynshield_berkeley()
