# -*- coding: utf-8 -*-
"""
Copyright (C) Koos C. J. Zevenhoven. All rights reserved.

Copying, modifying or distributing this library or sourcecode, 
via any medium, without permission from the author, is strictly 
prohibited.

This file was created on Tue May 14 17:01:20 2013

PyFunK is a library to create, store and manipulate single variable functions
such as time courses. The library was originally motivated by sequencing needs
for things like magnetic resonance imaging and spectroscopy as well as 
animation editing, which all require defining time courses and event timing. I 
expect this to be useful in a variety of contexts.

@author: Koos Zevenhoven
"""
from __future__ import unicode_literals, division, print_function

from calc_eddy_currents_for_patrik.tags import *
#from tags import *
import numpy as np

# TODO force fill values on left and right side (i.e. constant outside bounds)
# clipping will then also change the waveform outside the bounds, but new
# fill values may be specified
#
# UPDATE: already did this for ClipWaveform, but needs some more thoughts
# HOW SHOULD BOUNDS BE INTERPRETED LOGICALLY?
# Should all waveforms have some kind of fill value information and options
# for periodicity?

# TODO one big question: how about evaluating and clipping the waveform using
# slices? How does this relate to possibly array-centric cases?

# TODO another big question: what to do with tags?
# rewrite the whole tag thing? should there be tag pools?
# should tags just be instances of Constant (a subclass?)

# TODO desing plotting logic
# it is convenient to plot using an instance method, but how to deal with
# different plotting libraries and settings?


class Waveform(object):
    """An abstract base class for PyFunk functions."""    
    
    def __init__(self, defining_tags = ()):
        """Basic initializations.
        
        A subclass can override some of these instance variables using
        properties.
        """
        self.public_tags = tuple(defining_tags)
        self.bound_tags = (MINUS_INF, PLUS_INF) 
        self.private_tags = ()
    
    max_sampling_interval = .1e-3 #default TODO make more elegant
    
    def sample(self, positions):
        """Get a sampled version of the function at the given positions.
        
        Either this or value_at (or both) must be overridden by subclass.
        
        Arguments:
            positions -- a numpy vector (1d-array) of points in time
        """
        
        return np.array([self.value_at(p) for p in positions])

    def value_at(self, position):
        """Get a single sample at position/instant position.
        
        Either this or sample (or both) must be overriden by subclass.
        """
        return self.sample(np.array([float(position)]))[0]
    
    def is_zero(self):
        """Return True if Waveform is guaranteed zero in this context."""
        return False #default
    
    def is_const(self):
        """Return True if Waveform is guaranteed constant in this context."""
        return False #default
    
    def is_linear(self):
        """Return True if Waveform is guaranteed linear in this context."""
        return False #default
    
    @property
    def duration(self):
        """Return the duration of the waveform."""
        return self.bound_tags[1].value - self.bound_tags[0].value
    
    def as_piecewise(self):
        return PiecewiseWaveform([self])
#    def as_piecewise(self):
#        """Return this waveform represented as a PiecewiseWaveform."""
#        raise NotImplementedError("Cannot convert waveform to piecewise" + \
#                    "Type:" + type(self).__name__)
#    
    def to_samples(self, delta_t = None):
        """Return a Samples waveform representing the current state of self.
        
        NOTE: as_samples is deprecated, use to_samples instead (to emphasize
            that to_samples evaluates self at its current state when called)        
        
        Arguments:
            delta_t -- desired sampling interval (reproduced approximately!)
                When this is None, self.max_sampling_interval is used.
        """
        if delta_t is None:
            delta_t = self.max_sampling_interval
            
        t = np.linspace(self.bound_tags[0].value, self.bound_tags[1].value,
                        int(round(self.duration / delta_t + 1)))
        return Samples(self.sample(t), delta_t, float(self.bound_tags[0]))
    
    as_samples = to_samples
    
    def plot(self, *args, **kwargs):
        """Plot the waveform (temporary solution with matplotlib)"""
        from matplotlib import pyplot as plt
        t = np.linspace(float(self.bound_tags[0]), float(self.bound_tags[1]),
                        int(self.duration/1e-4))
        return plt.plot(t, self.sample(t), *args, **kwargs)
    
    def clip(self, left_bound = 0.0, right_bound = None,
             fill_values = (None, None)):
        """Return a clipped version of this waveform.
        
        The reference implementation is using ClipWaveform, but subclasses may
        override this behavior if something better is available.
        
        Arguments:
            left_bound -- new lower bound (or None for unchanged)
            right_bound -- new upper bound (or None for unchanged)
            fill_values -- values to fill with outside range (None->continuous)
        """
        return ClipWaveform(self, (left_bound, right_bound), fill_values)
    
    def delay(self, delay_time):
        """Return a delayed version of this waveform.
        
        The reference implementation is using DelayWaveform, but subclasses may
        override this behavior if something better is available.
        
        Arguments:
            delay_time -- the amount of shift in "time"
        """
        return DelayWaveform(self, delay_time)

    def convolve(self, other):
        """Return convolution with other.
        
        The reference implementation is using Convolution, but subclasses may
        override this behavior if something better is available.
        """
        return Convolution(self, other)
    
    def integrate(self):
        """Return integral of this waveform.

        The reference implementation is using Integral, but subclasses may
        override this behavior if something better is available.
        """
        return Integral(self)
    
    get_integral = integrate
    
    def differentiate(self):
        """Return derivative of this waveform.
        
        The reference implementation is using Integral, but subclasses may
        override this behavior if something better is available.
        """
        return Derivative(self)
    
    get_derivative = differentiate
    
    def def_integral(self, bounds = (None, None)):
        """Return the definite integral over the given interval.
        
        If None is given as a bound, the corresponding bound of the
        waveform itself is used.
        """
        #TODO: what happens if non-zero outside bounds?!
        #Default implementation: delegate to Samples
        return self.clip(*bounds).to_samples().def_integral()
        
    
    def append(self, other):
        return PiecewiseWaveform([self, other])
        
    def __neg__(self):
        return Unary(self, lambda x: -x)
    
    def __add__(self, other):
        try:
            float(other)
        except TypeError:
            if isinstance(other, Waveform):
                return NAry([self, other], lambda x, y: x + y)
            raise NotImplementedError()
        return Unary(self, lambda x: x + float(other))
        
    
    def __mul__(self, other):
        try:
            float(other)
        except ValueError:
            raise NotImplementedError()
        return Unary(self, lambda x: x * float(other))
    
    def __div__(self, other):
        try:
            float(other)
        except ValueError:
            raise NotImplementedError()
        return Unary(self, lambda x: x / float(other))

    __truediv__ = __div__    
    
    def __sub__(self, other):
        try:
            float(other)
        except ValueError:
            raise NotImplementedError()
        return Unary(self, lambda x: x - float(other))
        
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        try:
            float(other)
        except ValueError:
            raise NotImplementedError()
        return Unary(self, lambda x: -x + float(other))
    
    def __rdiv__(self, other):
        try:
            float(other)
        except ValueError:
            raise NotImplementedError()
        return Unary(self, lambda x: float(other) / x)    
    
    __rtruediv__ = __rdiv__
    
    def __pow__(self, other):
        try:
            float(other)
        except ValueError:
            raise NotImplementedError()
        return Unary(self, lambda x: x**float(other))
    
    def __rpow__(self, other):
        try:
            float(other)
        except ValueError:
            raise NotImplementedError()
        return Unary(self, lambda x: float(other)**x)
    
    def min(self):
        """Return minimum value of the function.
        
        This base class default implementation converts the waveform into
        samples and selects the smallest value, which may not be accurately
        the true minimum.
        """
        return self.to_samples().min()
    
    def max(self):
        """Return maximum value of the function.
        
        This base class default implementation converts the waveform into
        samples and selects the largest value, which may not be accurately
        the true maximum.
        """
        return self.to_samples().max()
    
    def mean(self):
        """Return mean value of the function.
        
        This base class default implementation converts the waveform into
        samples and takes the sample mean, which may not be accurately
        the true mean.
        """
        return self.to_samples().mean()
        
    def abs(self):
        """Return absolute value of the function."""
        return Unary(self, np.abs)
        
        
class Function(Waveform):
    """A waveform described by a given function (of "time")"""
    
    def __init__(self, function, bounds = (MINUS_INF, PLUS_INF),
                 fill_values = (0, 0)):
        """Create a waveform with values given by function(timepoints).
        
        Keep in mind that the function must be vectorized, so implement it
        assuming a numpy array of time points.
        """
        self._function = function
        self.bound_tags = tuple(ensure_tag(b) for b in bounds)
        self._fill_values = fill_values
        
        
    @property
    def function(self):
        return self._function
        
    def sample(self, positions):
        ret = np.empty(positions.shape)
        ret[positions < float(self.bound_tags[0])] = self._fill_values[0]
        ret[positions > float(self.bound_tags[1])] = self._fill_values[1]
        func_indices = (positions >= float(self.bound_tags[0])) \
                       * (positions <= float(self.bound_tags[1]))
        ret[func_indices] = self._function(positions[func_indices])
        return ret
        
class Time(Function):
    """A "waveform" whose value is simply that of "time"."""
    def __init__(self):
        Function.__init__(self, lambda x: x)

TIME = Time()     

class Unary(Waveform):
    """A waveform derived from another one through a pointwise function."""
    
    def __init__(self, waveform, function):
        """Create a derived waveform with values transformed by function.
        
        Arguments:
            waveform: a waveform to derive this one from
            function: a function to transform the values (accepts numpy array)
        """
        self._waveform = waveform
        self._function = function
        self.bound_tags = waveform.bound_tags
    
    @property
    def max_sampling_interval(self):
        return self._waveform.max_sampling_interval #not really optimal
    
    @property
    def function(self):
        return self._function
        
    def sample(self, positions):
        return self._function(self._waveform.sample(positions))
    
class NAry(Waveform):
    """EXPERIMENTAL FUNCTION OF N WAVEFORMS.
    
    FOR NOW, EVERYTHING OUT OF BOUNDS OF ANY OF THE WAVEFORMS WILL BE ZERO.
    """
    def __init__(self, waveforms, function):
        """EXPERIMENTAL FUNCTION OF N WAVEFORMS.
    
        FOR NOW, EVERYTHING OUT OF BOUNDS OF ANY OF THE WAVEFORMS WILL BE ZERO.
        """
        self._waveforms = list(waveforms)
        self._function = function
        
        l_tag = max((w.bound_tags[0] for w in self._waveforms), key = float)
        r_tag = min((w.bound_tags[1] for w in self._waveforms), key = float)
        self.bound_tags = (l_tag, r_tag)

    @property
    def max_sampling_interval(self):
        return min(wf.max_sampling_interval for wf in self._waveforms) #not really optimal
    
    def sample(self, positions):
        lout = positions < float(self.bound_tags[0])
        rout = positions > float(self.bound_tags[1])
        within = np.logical_not(np.logical_or(lout, rout))
        ret = np.empty(positions.shape)
        ret[within] = \
            self._function(*(w.sample(positions[within]) 
                                  for w in self._waveforms))
        ret[lout] = 0.0
        ret[rout] = 0.0
        return ret


class DelayWaveform(Waveform):
    """Reproduces another waveform with a delay (a shift in "time")."""
    
    def __init__(self, waveform, delay_time):
        """Create a delayed waveform, negative delay allowed."""
        self.waveform = waveform
        self._delay_time = delay_time
        tags = tuple(SumTag(tag.name, tag, delay_time)
                         for tag in waveform.bound_tags)
        Waveform.__init__(self, defining_tags = tags)
        self.bound_tags = tuple(tags)
        
    @property
    def max_sampling_interval(self):
        return self.waveform.max_sampling_interval
        
    def is_zero(self):
        return self.waveform.is_zero()
    
    def is_const(self):
        return self.waveform.is_const()
    
    def is_linear(self):
        return self.waveform.is_linear()
    
    def sample(self, positions):
        return self.waveform.sample(positions - self._delay_time)
        
    
class ClipWaveform(Waveform):
    """Clips a waveform into new bounds."""
    
    def __init__(self, waveform, new_bounds = (ZERO, None),
                 fill_values = (None, None)):
        """Makes a view waveform clipped to the new bounds.
        
        None as a bound means take the original bound.
        
        None as fill value makes the function continuous at bounds, i.e.
        fills with value at bound.
        """
        try:
            pubtags = waveform.public_tags
        except:
            pubtags = ()
        Waveform.__init__(self, pubtags)
        bounds = list(new_bounds)
        for i in range(2):
            if bounds[i] == None:
                bounds[i] = waveform.bound_tags[i]
            else:
                bounds[i] = ensure_tag(bounds[i])
        
        self.bound_tags = tuple(bounds)
        
        self._fill_values = fill_values
        self._waveform = waveform

# TODO make sure these are ok
#    def is_zero(self):
#        return self._waveform.is_zero()
#    
#    def is_const(self):
#        return self._waveform.is_const()
#    
#    def is_linear(self):
#        return self._waveform.is_linear()
#    
    @property
    def max_sampling_interval(self):
        return self._waveform.max_sampling_interval
    
    def sample(self, positions):
        smp = self._waveform.sample(positions)

        if self._fill_values[0] is None:
            smp[positions < self.bound_tags[0].value] = \
                self._waveform.value_at(self.bound_tags[0].value)
        else:
            smp[positions < self.bound_tags[0].value] = self._fill_values[0]
        if self._fill_values[1] is None:
            smp[positions > self.bound_tags[1].value] = \
                self._waveform.value_at(self.bound_tags[1].value)
        else:
            smp[positions > self.bound_tags[1].value] = self._fill_values[1]
        
        return smp
            
        
class Derivative(Waveform):
    """Derivative of a waveform."""
    
    def __init__(self, waveform):
        """Make derivative waveform of waveform."""
        self.waveform = waveform
        self.bound_tags = self.waveform.bound_tags

    @property
    def max_sampling_interval(self):
        return self.waveform.max_sampling_interval
    
    def sample(self, positions):
        dt = self.waveform.max_sampling_interval/2
        t_right = positions + dt/2       
        t_left = positions - dt/2
        smp_right = self.waveform.sample(t_right)
        smp_left = self.waveform.sample(t_left)
        diff = (smp_right - smp_left) / dt
        
        # fix endpoints (assume possibly discontinuous)
        # TODO fix also other cases, discontinuities etc?)
        
        def right_derivative(where):
            return (smp_right[where] - \
                    self.waveform.sample(positions[where])) / (dt/2)
        def left_derivative(where):
            return (self.waveform.sample(positions[where]) - \
                    smp_left[where]) / (dt/2)
        
        # first case: point in range (left) but corresponding t_left is not
        l_low = t_left < self.waveform.bound_tags[0].value
        diff[l_low] = right_derivative(l_low)
        
        # second case: point itself not in range (OVERRIDES FIRST CASE)
        c_low = positions < self.waveform.bound_tags[0].value
        diff[c_low] = left_derivative(c_low)
        
        # similar two cases at the right boundary
        r_high = t_right > self.waveform.bound_tags[1].value
        diff[r_high] = left_derivative(r_high)
        # second case, again, overrides when necessary
        c_high = positions > self.waveform.bound_tags[1].value
        diff[c_high] = right_derivative(c_high)
        
        return diff
        

class Integral(Waveform):
    """Integral of a waveform."""
    
    def __init__(self, waveform):
        """Make integral waveform of waveform."""
        self.waveform = waveform
        self.bound_tags = self.waveform.bound_tags
    
    @property
    def max_sampling_interval(self):
        return self.waveform.max_sampling_interval

    def sample(self, positions):
        from scipy import integrate
        #TODO handle infinite waveforms'
        dt = float(self.max_sampling_interval)
        t = np.arange(float(self.waveform.bound_tags[0]),
                      float(self.waveform.bound_tags[1]) + 2*dt, dt)
        smp = self.waveform.sample(t)
        i_smp = integrate.cumtrapz(smp, x=t, dx=dt)
        return Samples(i_smp, dt, 
                       self.waveform.bound_tags[0]).sample(positions)
        
        
class Convolution(Waveform):
    """A convolution of two waveforms.
    
    TODO Take care of problems arising if the waveform is not zero outside its 
    range. Also has to do with end effects of truncation looking like a step.
    """
    
    def __init__(self, waveform, kernel):
        """Make convolution of waveform with kernel."""
        self.waveform = waveform
        self.kernel = kernel
        
        self.bound_tags = (SumTag("left_bound", 
                                  waveform.bound_tags[0],
                                  kernel.bound_tags[0]),
                           SumTag("right_bound",
                                  waveform.bound_tags[1],
                                  kernel.bound_tags[1]))
    
    @property
    def max_sampling_interval(self):
        return min(float(self.waveform.max_sampling_interval),
                   float(self.kernel.max_sampling_interval))
    
    def value_at(self, position):
        import scipy.integrate as integrate
        #TODO optimize this using simps, trapz etc. and for special cases
        # of waveform types
        t = float(position)        

        wave = self.waveform.value_at
        kern = self.kernel.value_at
        def integrand(tau):
            return wave(t - tau) * kern(tau)
            
        i = integrate.quad(integrand, 
                           float(self.kernel.bound_tags[0]),
                           float(self.kernel.bound_tags[1]))[0] 
        return i
    
    def sample(self, positions):
        dt = self.max_sampling_interval
        
        w_range = np.arange(float(self.waveform.bound_tags[0]), 
                            float(self.waveform.bound_tags[1]) + dt, dt)
        k_range = np.arange(float(self.kernel.bound_tags[0]), 
                            float(self.kernel.bound_tags[1]) + dt, dt)
        
        w = self.waveform.sample(w_range)
        k = self.kernel.sample(k_range)
        
        if not hasattr(self, '_asSamples'):
            c = np.convolve(w, k, mode = "full") * dt
            samples = Samples(c, dt, self.bound_tags[0])
            self._asSamples = samples.clip(*(t.value for t in self.bound_tags))    
        return self._asSamples.sample(positions)
        
    
class Constant(Waveform):
    """A constant of a given duration, which can also be zero."""
    
    def __init__(self, value, start = MINUS_INF, end = PLUS_INF):
        """Create a constant waveform."""
        
        start = ensure_tag(start)
        end = ensure_tag(end)
        
        super(Constant, self).__init__((start,end))
        self.bound_tags = tuple(self.public_tags)
        self._value = value
        self.max_sampling_interval = float('inf')
        
    def value_at(self, instant):
        """Return value at t=instant.
        
        instant can also be a Tag."""
        
        return float(self._value)
    
    def sample(self, positions):
        return np.ones(positions.shape) * float(self._value)
        
    def is_const(self):
        return True
    
    def is_linear(self):
        return True
        
    def is_zero(self):
        if self._value == 0:
            return True
        return False

class Zero(Constant):
    """A zero-valued constant."""
    
    def __init__(self, interval = (MINUS_INF,PLUS_INF)):
        super(Zero, self).__init__(0, interval)


class Ramp(Waveform):
    """An abstract class for a ramp.
    
    A ramp can be used in a SequenceFunction between two other blocks of
    the sequence to interpolate between two values.
    """
    
    def __init__(self, start_time_and_value, end_time_and_value):
        """Set up functions on the left and right."""
        (start, start_value) = start_time_and_value
        (end, end_value) = end_time_and_value
        start = ensure_tag(start)
        end = ensure_tag(end)
        
        super(Ramp, self).__init__((start, end))
        self.bound_tags = self.public_tags
        
        self.start_value = start_value
        self.end_value = end_value
    
    def _sample_when_zero_duration(self, positions):
        """For subclasses to call if ramp to be sampled has zero duration.
        
        Anything < the starting time will get the start value and anything
        >= the starting time will get the end value. Of course start time
        and end time are probably equal when this is called.
        """

        step_pos = self.bound_tags[0].value
        values = np.empty(positions.shape)
        values[positions < step_pos] = float(self.start_value)
        values[positions >= step_pos] = float(self.end_value)
        return values
    
    def _to_zero_to_one(self, positions):
        factor = (positions - float(self.bound_tags[0])) / self.duration
        
        factor[factor < 0.0] = 0.0
        factor[factor > 1.0] = 1.0
        
        return factor

class LinearRamp(Ramp):
    """A linear (affine) function, i.e. a line."""
    
    def is_linear(self):
        return True
    
    def is_zero(self):
        if self.start_value == 0 and self.end_value == 0:
            return True
        return False
    
    def is_constant(self):
        if self.start_value == self.end_value:
            return True
        return False
        
    def sample(self, positions):
        if self.duration == 0:
            return self._sample_when_zero_duration(positions)
            
        factor = self._to_zero_to_one(positions)
        s = float(self.start_value)
        return factor * (float(self.end_value) - s) + s
    
    def min(self):
        return min(float(self.start_value), float(self.end_value))
    
    def max(self):
        return max(float(self.start_value), float(self.end_value))

class HalfSineRamp(Ramp):
    """A smooth ramp with the shape of sin(x), x = -pi/2..pi/2."""
    def sample(self, positions):
        if self.duration == 0:
            return self._sample_when_zero_duration(positions)
        
        factor = np.cos(self._to_zero_to_one(positions) * np.pi) * -0.5 + 0.5
        factor[positions > float(self.bound_tags[1])] = 1.0
        s = float(self.start_value)
        
        return factor * (float(self.end_value) - s) + s
    
    def min(self):
        return min(float(self.start_value), float(self.end_value))
    
    def max(self):
        return max(float(self.start_value), float(self.end_value))

class QuarterCosineRamp(Ramp):
    """A smooth ramp with the shape of cos(x), x = 0..pi/2."""
    def sample(self, positions):
        if self.duration == 0:
            return self._sample_when_zero_duration(positions)
        
        factor = np.cos(self._to_zero_to_one(positions) * np.pi/2) * -1 + 1
        factor[positions > float(self.bound_tags[1])] = 1.0
        s = float(self.start_value)
        
        return factor * (float(self.end_value) - s) + s
    
    def min(self):
        return min(float(self.start_value), float(self.end_value))
    
    def max(self):
        return max(float(self.start_value), float(self.end_value))
    
     
class PiecewiseWaveform(Waveform):
    """Describes a function composed of 'sequential' blocks.   

    In mathematics, this is comparable with a piecewise defined function,
    but is more flexible. At points in time that are out of bounds for
    all the waveform blocks, the next waveform in time will determine the 
    value (possibly as a constant equal to the start value). If two waveform 
    blocks overlap, the one that begins earlier will determine the value of the
    sequence in the overlapping region. If two blocks begin at the same time,
    the one that ends earlier will be prioritized. After the last block, the
    last waveform determines the value.
    """
    
    def __init__(self, blocks = []):
        """Create a sequence of waveforms, empty by default."""
        self.blocks = blocks
    
    def append(self, block):
        """Append a waveform block at the end of the sequence."""
        
        if isinstance(block, PiecewiseWaveform):
            for b in block.blocks:
                self.append(b)
                return self
        
        self.blocks.append(block)
        return self
        
    
    def get_sorted_blocks(self):
        if len(self.blocks) == 0:
            return
            
        #first sort according to secondary criterion: the end tag
        sblocks = sorted(self.blocks, key=lambda wf: wf.bound_tags[1].value)
        #then sort the blocks according to the start tags (rely on stability)
        sblocks = sorted(sblocks, key=lambda wf: wf.bound_tags[0].value)
        
        
        #find start and end tag for the active period of each block
        start_active = sblocks[0].bound_tags[0]
        for b in sblocks[:-1]:
            end_active = b.bound_tags[1]
            yield (b, start_active, end_active)
            start_active = end_active
        
        yield (sblocks[-1], start_active, sblocks[-1].bound_tags[1])              
        
    
    def sample(self, positions):
        if positions.size == 0:
            return positions.copy()
        values = np.empty(positions.shape)  
        
        lowbound = min(positions)
        for wf, start_active, end_active in self.get_sorted_blocks():
            #TODO handle values that are out of bounds
            bounds = [tag.value for tag in wf.bound_tags]
            indices = (positions >= lowbound) * (positions < bounds[1])
            
            values[indices] = wf.sample(positions[indices])
            
            lowbound = bounds[1]
        
        # handle remaining positions
        indices = positions >= lowbound
        values[indices] = wf.sample(positions[indices])
        
        return values
            
    @property
    def bound_tags(self):
        # bounds: low bound of first block in time and high bound of 
        # last
        firstblock = min(self.blocks, key = lambda wf: wf.bound_tags[0].value)
        lastblock = max(self.blocks, key = lambda wf: wf.bound_tags[1].value)
        return (firstblock.bound_tags[0], lastblock.bound_tags[1])
    
    def is_zero(self):
        return all(b.is_zero() for b in self.blocks)
    
    def as_piecewise(self):
        return self
    
    def delay(self, delay_time):
        return PiecewiseWaveform([b.delay(delay_time) for b in self.blocks])

class MultiRamp(PiecewiseWaveform):
    """Describes multiple constant levels connected with ramps."""
    
    def __init__(self,
                 start_positions,
                 start_values, 
                 ramp_times, 
                 ramp_types, first_value = 0.0):
        pass
    

class RampedBox(PiecewiseWaveform):
    """Describes a box pulse with given ramp types."""
    #TODO extend this from MultiRamp
    
    def __init__(self, amp, start_end_tuple, rise_fall_tuple,
                 ramp_types = (LinearRamp,) * 2,
                 ramp_params = (tuple(),) * 2,
                 final_value = 0.0):
        """Create a ramped box pulse starting at tag start and end.
        
        Arguments:
            amp -- amplitude of pulse
            start_end_tuple -- (start_time, end_time)
                start_time -- starting time tag
                end_time -- ending time tag
            rise_fall_tuple -- (rise_time, fall_time)
                rise_time -- duration of ramp-up
                fall_time -- duration of ramp-down
            ramp_types -- tuple of Ramp subclasses for start and end
            ramp_params -- tuple of two tuples of additional parameters for
                           ramp class constructor after defining the start and
                           end points
        """
        (start_time, end_time) = start_end_tuple
        (rise_time, fall_time) = rise_fall_tuple
        start_time = ensure_tag(start_time)
        end_time = ensure_tag(end_time)
        rise_time = ensure_tag(rise_time)
        fall_time = ensure_tag(fall_time)
        amp = ensure_tag(amp)
        self._amp = amp
        
        ramp1_bounds = (Tag("start_rampup", start_time),
                        SumTag("end_rampup", start_time, rise_time))
        ramp2_bounds = (Tag("start_rampdown", end_time),
                        SumTag("end_rampdown", end_time, fall_time))
        
        ramp1 = ramp_types[0]((ramp1_bounds[0], 0), 
                              (ramp1_bounds[1], amp),
                              *ramp_params[0])
        ramp2 = ramp_types[1]((ramp2_bounds[0], amp), 
                              (ramp2_bounds[1], final_value),
                              *ramp_params[1])
        
        const = Constant(amp, 
                         float(ramp1.bound_tags[1]), 
                         float(ramp2.bound_tags[0]))
        #TODO think about how the end should be handled
        #this all has to do with what waveforms can do outside their ranges
        endconst = Constant(0, 
                            float(ramp2.bound_tags[1]),
                            float(ramp2.bound_tags[1]))
        super(RampedBox, self).__init__([ramp1, const, ramp2, endconst])
    
    @property
    def amp(self):
        return self._amp

class Box(RampedBox):
    """A box pulse during a given time interval"""
    def __init__(self, start, end, *, amp=1):
        """Create a box pulse.
        
        Arguments:
            start = beginning of pulse (up ramp)
            end = end of pulse (down ramp)
            amp = amplitude (optional, default = 1)
        """
        # could use a more dummy ramp than linear ramp (Discontinuity?)
        super().__init__(amp, (start, end), (0,0))
        

class Samples(Waveform):
    """A basic sampled waveform with a given sampling interval.
    
    For now, it gives zero outside the sampled range (to be defined better).    
    """

    def __init__(self, samples, delta_t, first_sample_time = ZERO):
        """Create a sampled waveform.

        Arguments:
            samples -- samples at a fixed interval
            delta_t -- sampling interval
        """
        self._samples = samples
        self._delta_t = delta_t
        
        duration = (len(self._samples) - 1) * self._delta_t
        
        first_sample_tag = ensure_tag(first_sample_time)
        
        last_sample_tag = SumTag("last_sample", first_sample_tag, duration)
        self.bound_tags = (first_sample_tag, last_sample_tag)
    
    #TODO compare with to_samples!!! make this stuff consistent!
    def as_samples(self, delta_t = 0.1e-3):
        if self._delta_t == delta_t:
            return self
        else:
            return Waveform.as_samples(self, delta_t = delta_t)
            
            
    @property
    def time(self):
        return np.linspace(float(self.bound_tags[0]), float(self.bound_tags[1]),
                           self._samples.size)
        
        
    def _sample(self, positions, samples, fill_with_ends = (False, False)):
        from scipy import interpolate
       
        interp = interpolate.interp1d(self.time, samples, 
                                      kind='linear', copy = False, 
                                      bounds_error=False, fill_value=0.0)
        ret =  interp(positions)
        if fill_with_ends[0]:
            ret[positions <= float(self.bound_tags[0])] = samples[0]
        if fill_with_ends[1]:
            ret[positions >= float(self.bound_tags[1])] = samples[-1]
        
#        ret[positions < self.bound_tags[0].value] = samples[0]
#        ret[positions > self.bound_tags[1].value] = samples[-1]
        return ret
    
    def sample(self, positions):
        #this put in _sample() to help out subclassese
        return self._sample(positions, self._samples)
        
    def add_samples(self, other):
        #TODO extend beyond limits if needed?
		
        return Samples(self._samples + other.sample(self.time), self._delta_t)
    
    def neg_samples(self):
        return Samples(-self._samples, self._delta_t)
    
    def mul_samples(self, other):
        if isinstance(other, Waveform):
            other = other.sample(self.time)
        return Samples(self._samples * other, self._delta_t)
    
    def div_samples(self, other):
        if isinstance(other, Waveform):
            other = other.sample(self.time)
        return Samples(self._samples / other, self._delta_t)
    
    def sub_samples(self, other):
        return self + (-other)
        
#    def radd__(self, other):
#        return self + other
    
#    def rmul__(self, other):
#        return self * other
    
#    def rsub__(self, other):
#        return -self + other
    
    def min(self):
        return self._samples.min()
    
    def max(self):
        return self._samples.max()
        
    def mean(self):
        return self._samples.mean()
    
    def def_integral(self, bounds = (None, None)):
        from scipy import integrate   
        if bounds == (None, None):
            return integrate.trapz(self._samples, self.time)
        else:
            #This could be done without resampling
            return self.clip(*bounds).def_integral()


def make_SampledTailRamp(samples, delta_t = None):
    """Make a "ramp" class described by the given samples.

    The samples should be represent a ramp from 0 to 1.    
    
    Arguments:
        samples -- an array of samples or a Samples instance
        delta_t -- sampling interval if array of samples given
    """
    if isinstance(samples, Waveform):
        if not isinstance(samples, Samples):
            samples = samples.as_samples()
        delta_t = samples._delta_t        
        samples = samples._samples
        
    # TODO derive this also from Ramp and make sure out-of-range values
    # are "0" and "1" (STANDARDIZE ALL OF THIS BETTER IN THE MODULE!)
    class SampledTailRamp(Samples):
        """A "ramp" described by a sampled waveform.
        
        The "ramp" is defined by samples, and may extend past the given end 
        tag.
        """
        ramp_samples = samples.copy()
        def __init__(self, start_time_and_value, end_time_and_value):
            
            (start, start_value) = start_time_and_value
            (end, end_value) = end_time_and_value
            Samples.__init__(self, 
                             self.ramp_samples, delta_t, ensure_tag(start))
        
            self.start_value = start_value
            self.end_value = end_value
        
        def sample(self, positions):
            step = float(self.end_value) - float(self.start_value)
            samples = self._samples * step + float(self.start_value)
            return self._sample(positions, samples)
        
    return SampledTailRamp
        
if __name__ == "__main__":
    box = RampedBox(1,(10,20), (0,2))
    t = np.linspace(0,30,300)
    y = box.sample(t)
    import pylab    
    pylab.plot(t,y)
    pylab.show()
    
    from .seqtools import Sequence
    seq = Sequence()
    seq["chan"] = box
    from .seqgencode import SequenceEncoder
    enc = SequenceEncoder()
    
    
