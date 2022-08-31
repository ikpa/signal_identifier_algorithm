# -*- coding: utf-8 -*-
"""
Copyright (C) Koos C. J. Zevenhoven. All rights reserved.

Copying, modifying or distributing this library or sourcecode, 
via any medium, without permission from the author, is strictly 
prohibited.

This file was created on Wed Jul 10 16:05:29 2013

A python labrary for reading data from the MEG-MRI system measurement channels
in real time.

@author: Koos Zevenhoven
"""
from __future__ import unicode_literals, division, print_function

import numpy as np
import threading
import multichan

_ACQUIRE_SFREQ = 10e3

class MultiChannelData(multichan.ChannelPool):
    """Represents sampled signals with named channels."""
    
    def __init__(self, data, channel_names, 
                 sampling_frequency = _ACQUIRE_SFREQ, 
                 start_time = 0.0):
        """Make a data set from given data and channel names.
        
        Arguments:
            data -- a 2d numpy array, each column representing a channel
            channel_names -- an iterable of the names of the channels in order
            sampling_frequency -- the sampling frequency
            start_time -- time at the first sample
            
        """  
        super_ = super(MultiChannelData, self)
        self.data = data
        self.names = list(channel_names)
        self.n_channels = data.shape[1]
        self.n_samples = data.shape[0] 
        self.sampling_rate = sampling_frequency
        super_.__init__(((self.names[i], data[:,i]) 
                              for i in range(self.n_channels)))
        
        end_time = start_time + float(self.n_samples)/sampling_frequency
        self.time = np.linspace(start_time, 
                                end_time,
                                self.n_samples,
                                endpoint = False)
                                
                                
    def save_npz(self, target_file):
        """Save the data using a numpy npz file.
        
        Example:
        
        # Save the MultiChannelData object to file "diskfile.npz"
        data.save_npz("diskfile.npz")
        
        # Load the data back into data2, which should then look like data
        data2 = MultiChannelData.load_npz("diskfile.npz")
        
        Arguments:
            target_file: file name or file object to save to
        """
        from numpy import savez

        descr = """Multichannel 1-D data in a 2-D array.
        Each column in "data" represents one channel. The names of the 
        channels are given by "names". The time values are given by "time"."""
        
        d = dict(descr = descr,
                 data = self.data,
                 names = self.names,
                 time = self.time)
        
        savez(target_file, **d)
                    
    @staticmethod
    def load_npz(source_file):
        """Load data from a numpy npz file (saved by .save_npz()).
        
        Example:
        
        # Save the MultiChannelData object to file "diskfile.npz"
        data.save_npz("diskfile.npz")
        
        # Load the data back into data2, which should then look like data
        data2 = MultiChannelData.load_npz("diskfile.npz")
        
        Arguments:
            source_file: file name or file object to load from
        
        Return:
            MultiChannelData object with the data from file
        """
        from numpy import load
        
        d = load(source_file)
        
        return MultiChannelData(d["data"], d["names"], 
                                1.0/(d["time"][1] - d["time"][0]),
                                d["time"][0])
        
    def clip(self, *time_window):
        """Return a specified time window of the data.
        
        The returned MultiChannelData object will be a restricted 
        view into self. In almost all cases, samples will be from the 
        interval [t1, t2). In very rare cases where the time window length
        is very slightly smaller than an integer multiple of sampling
        intervals, one of the taken sampling instants may be outside the range
        by a very small amount.
        
        It is ensured that for a fixed interval length t2 - t1 (and fixed
        sampling rate), the same number of sampling instants is always 
        included.        
        
        Calling signatures:
            .clip(10e-3, 20e-3) : clip to between time = 10e-3 and 20e-3
            .clip((10e-3, 20e-3)) : : clip to between time = 10e-3 and 20e-3
        
        Return:
            MultiChannelData object clipped to the given interval.
        """
        try:
            t1 = float(time_window[0])
            t2 = float(time_window[1])
        except TypeError:
            time_window = time_window[0] #maybe the window is a tuple
            try:
                t1 = float(time_window[0])
                t2 = float(time_window[1])
            except TypeError:
                raise
        
        fp_margin = 1e-9
        
        n = np.floor((t2 - t1 + fp_margin) * self.sampling_rate)
        indices = self.time >= t1 - fp_margin
        try:        
            first_t = self.time[indices][0]
            indices *= self.time < first_t + (n - 0.5) / self.sampling_rate
                
            t0 = self.time[indices][0]
        except IndexError:
            raise ValueError("No data points in clipping range")
        
        return MultiChannelData(self.data[indices,:],
                                self.names,
                                self.sampling_rate,
                                t0)
    
    def subpool(self, chan_list):
        
        chans = list(self.match(chan_list))
        
        new_names = [n for n in self.names if n in chans]
        chan_mask = [n in chans for n in self.names]
        
                
        return MultiChannelData(self.data[:,np.array(chan_mask)],
                                new_names,
                                self.sampling_rate,
                                self.time[0])
    subpool.__doc__ = multichan.ChannelPool.subpool.__doc__
    
    def __add__(self, other):
        if any(n1 != n2 for n1, n2 in zip(self.names, other.names)):
            raise Exception("Should have the same channels for summing")
        return MultiChannelData(self.data + other.data, self.names,
                                sampling_frequency = self.sampling_rate,
                                start_time = self.time[0])
    
    def __mul__(self, factor):
        return MultiChannelData(self.data * factor, self.names,
                                sampling_frequency = self.sampling_rate,
                                start_time = self.time[0])
    
    def __truediv__(self, value):
        return self * (1.0/value)
        
     
    def plot(self, chans = "*"):
        """Plot channels (all by default)."""        
              
        import pylab as pl
        
        chans = list(self.match(chans))
        if len(chans) > 8:
            import warnings
            warnings.warn("Plotting so many channels is not recommended now.")
        
        for i, chan in enumerate(chans):
            pl.subplot(len(chans), 1, i+1)
            pl.plot(self.time, self[chan].flatten())
            pl.ylabel(chan)
        pl.xlabel("Time [s]")
        
    def fft_all(self):
        """Return the fft (complex) of the data in all channels.

        The DC offset is removed and the data is windowed (blackman) 
        before FFT. The FFT is scaled so that its abs()**2 is the spectral
        density.
        
        Returns:
            a dictionary-like of FFTs, which also has properties .freq (the
            frequencies corresponding to the FFT points) and .data (FFTs of
            all channels in one 2-D array)
        """

        freq = np.fft.fftfreq(self.data.shape[0], 1./self.sampling_rate)
        freq = np.fft.fftshift(freq, axes = (0,))

        data = self.data - self.data.mean(axis = 0)
        
        window = np.blackman(data.shape[0])
        window /= window.sum()*np.sqrt(np.abs((freq[1]-freq[0])))
        data *= window.reshape((-1,1)) * np.sqrt(2)
        data = np.fft.fft(data, axis = 0)
        data = np.fft.fftshift(data, axes = (0,))
        
        
        pool = multichan.ChannelPool({self.names[i] : data[:,i] 
                                          for i in range(self.n_channels)})
        pool.data = data
        pool.freq = freq         
        return pool
        
    
    def fft(self, chan, start_and_end = (None, None)):
        """Return the fft of a channel within an interval"""
        start_and_end = list(start_and_end)
        if start_and_end[0] == None:
            start_and_end[0] = self.time[0]
        if start_and_end[1] == None:
            start_and_end[1] = self.time[-1]
            
        indices = (self.time >= start_and_end[0]) * \
                    (self.time <= start_and_end[1])
        freq = np.fft.fftfreq(np.sum(indices), 1./self.sampling_rate)
        
        data = self[chan].flatten()[indices] #flatten returns copy => safe
        data -= data.mean()
        window = np.blackman(data.size)
        window /= window.sum()*np.sqrt(np.abs((freq[1]-freq[0]))) 
        #TODO check this normalization (should multiply by sqrt(2) because of omitted half?) DONE
        
        data *= window * np.sqrt(2)
        data = np.fft.fft(data)
        return np.fft.fftshift(freq), np.fft.fftshift(data)
        
    def plot_fft(self, chans, interval_tuple, logy = True):
        """Plot the fft of a channel within an interval."""
        (t_start, t_end) = interval_tuple
        import pylab as pl
        
        chans = list(self.match(chans))
            
        #TODO use get_fft here?
        
        indices = (self.time >= t_start) * (self.time <= t_end)
        freq = np.fft.fftfreq(np.sum(indices), 1./self.sampling_rate)
        
        for i, chan in enumerate(chans):
            pl.subplot(len(chans), 1, i+1)
            data = self[chan].flatten()[indices] #flatten returns copy => safe
            data -= data.mean()
            window = np.blackman(data.size)
            window /= window.sum()*np.sqrt(np.abs((freq[1]-freq[0])))
            #TODO check this normalization (should multiply by sqrt(2) because of omitted half?)            
            data *= window * np.sqrt(2)
            data = np.fft.fft(data)
            if logy:
                pl.semilogy(freq, np.abs(data))
                pl.ylim(7e-16,4e-12)
                pl.xlim(0,4000)

            else:
                pl.plot(freq, np.abs(data))
                pl.ylim(0,4e-13)
                pl.xlim(2000,3000)

            pl.ylabel(chan)
        pl.xlabel("Freq [Hz]")
    

        
def load_fiff(filename, channels = None, index_range = None):
    """Load a fiff file into MultiChannelData."""
    from mne.fiff import Raw
    raw = Raw(filename)
        
    if index_range == None:
        index_range = (0, raw.n_times)
    
    if channels != None:
        channels = list(multichan.wildcardkeys(raw.ch_names, channels))
        n_chans = len(channels)
    else:
        n_chans = len(raw.ch_names)
        channels = raw.ch_names
    
    data = np.empty((index_range[1] - index_range[0], n_chans))

    for i, ch in enumerate(channels):
        vals, t = raw[raw.ch_names.index(ch),index_range[0]:index_range[1]]
        data[:, i] = vals.ravel()
    
    ret = MultiChannelData(data, channels, 
                           sampling_frequency = raw.info["sfreq"])
    raw.close()
    return ret
        

class Trigger(object):
    """A simple trigger object for triggering based on signals."""
    RISING = 1
    FALLING = -1
    
    #TODO: buffer if delay negative
    #TODO: would it be possible not to use AQCUIRE_SFREQ as constant here
    def __init__(self, channel, trigger_level = 0.0, edge = RISING, 
                 trigger_time = 0.0, start_time = None, stop_time = None,
                 n_samples = None):
        """Create a trigger at given signal level on a given channel.
        
        Arguments:
            channel: name of channel to trigger on
            trigger_level: signal level at which to trigger
            edge: rising or falling edge (Trigger.RISING, Trigger.FALLING)
            trigger_time: time instant [s] to be fixed at the trigger point
            start_time: acquisition start time [s]
            stop_time: acquisition stop time [s]
            
            (time values are not used here, but are saved as attributes)
        """
        if n_samples != None and stop_time != None: 
            raise RuntimeError("Specify *either* n_samples or stop_time.")

        self.channel = channel
        self.level = trigger_level
        self.edge = edge
        self._ready = False
        self.trigger_time = trigger_time
        self.start_time = start_time if start_time != None else trigger_time
        if stop_time != None or n_samples != None:
            # I don't remember what this was needed for. It now assumes the
            # current default measurement sampling rate. Hmm..
            acquire_sfreq = get_measurement_sampling_rate()
            self.stop_time = stop_time if stop_time != None \
                else float(n_samples) / acquire_sfreq
            self.n_samples = n_samples if n_samples != None \
                else int((self.stop_time - self.start_time) 
                         * acquire_sfreq)
            
        #self.last_sample = 0.0
    
    def check(self, multichanneldata):
        """Check the given multichannel data for triggering.
        
        Return None if not triggered. If triggered, return the sample index 
        where the triggering occurs.
        """
        data = multichanneldata[self.channel].flatten()
        cond = data*self.edge > self.level*self.edge
        
        ready_i = 0
        if not self._ready: # first wait until condition is _not_ met
            ready_i = next((i for i in range(data.size) if not cond[i]), None)
            if ready_i == None:
                return None
            
            self._ready = True
        ret = next((i for i in range(ready_i, data.size) if cond[i]), None)

        if ret != None:
            self._ready = False
        
        return ret
        
        

def measure(n_samples = None, 
            duration = None,
            trigger = None, 
            port = None, 
            timeout_sec = 15):
    """Shortcut for measuring a n_samples samples of selected channels.
    
    Return a MultichannelData object.
    """
        
    thread = AcquireThread(n_samples, 
                           duration = duration,
                           trigger = trigger,
                           port = port,
                           timeout_sec = timeout_sec)
    thread.start()
    return thread.get_result()

    
def get_measurable_channels(chan_list = None, port = None):
    """Return list of names of channels that can be measured in realtime.
    
    If chan_list is given, only the channel names matching the list are
    returned (wildcards ok).
    """
    try:
        #Read dummy data (not absolutely necessary but doesn't hurt either)
        data = measure(2, port = port)
        if chan_list != None:
            return data.match(chan_list)
        return list(data.keys())
    except:
        return []

def get_measurement_sampling_rate(port = None):
    #Read dummy data (not absolutely necessary but doesn't hurt either)
    data = measure(2, port = port)
    return data.sampling_rate

class AcquireThread(threading.Thread):
    """Acquire a period of data in a separate thread.
    
    As an instance of threading.Thread, you shoud start it by calling start().
    """
    
    def __init__(self, n_samples = None, duration = None, trigger = None, 
                 port = None, timeout_sec = 5):
        """Create a thread object. Arguments like for acquire_period()."""
        from sequencer.daq_backends import fieldtrip
        self.data = None
        
        #for now, always initialize default backend here
        self.backend = fieldtrip.FieldTripMNEBackend(host = None,
                                                     port = port)
        
        def thread_fun():
            self.data = \
                self.backend.acquire_period(n_samples = n_samples,
                                            duration = duration,
                                            trigger = trigger,
                                            timeout_sec = timeout_sec)
        threading.Thread.__init__(self, target = thread_fun)
        
    
    @property
    def has_data(self):
        return self.data is not None
        
    def get_result(self):
        """Get the results (data, channel_names).
        
        This will first call join() on this thread.
        """
        self.join()
        if self.data == None:
            raise IOError("Realtime data acquisition was unsuccessful")
        return self.data

def _acquire_period(n_samples = None, duration = None, 
                    trigger = None, port = None, timeout_sec = 5):
    """Read a period of realtime data.
    
    This is mainly used from within AcquireThread. Use measure() instead.
    
    If a trigger is specified, this will wait for the trigger event to occur
    and return data based on that. If None, acquisition will start from
    the first available sample.
    
    Returns:
        data -- numpy array, each column being a channel
        channel_names -- names / labels of the channels
        
    """
    import FieldTrip
    
    if duration is not None and n_samples is not None:
        raise ValueError("Do not specify both duration and number of samples")
    if duration is None and n_samples is None:
        raise ValueError("Please specify either duration or n_samples")
    if n_samples is not None and not isinstance(n_samples, int):
        raise ValueError("Number of samples must be an integer")
    
    if trigger != None and trigger.start_time < trigger.trigger_time:
        raise NotImplementedError("Starting acquisition before trigger.")
    from time import time
    time_start = time()
    
    ft = FieldTrip.Client()             
    
    if port == None:
        port = REALTIME.PORT
        
    ft.connect(REALTIME.HOST, port)
    ret_buffer = None
    buf_index = 0
    channel_names = None
    start_index = 0
    
    #n_samples = int(n_samples) #Added conversion to int (hopefully okay)
    
    triggered = True if trigger == None else False
    
    while True:
        header = ft.getHeader()
        if header == None:
            raise IOError('Realtime header could not be read.')
            
        if n_samples is None:
            # duration has been given
            n_samples = int(round(duration * header.fSample))
            
        next_smp = header.nSamples
        next_evt = header.nEvents
        if channel_names != None:
            if any(c1 != c2 for c1, c2 in zip(channel_names, header.labels)):
                raise IOError('Realtime buffer header changed unexpectedly.')
        channel_names = header.labels
        if ret_buffer == None:
            ret_buffer = np.empty((n_samples, len(channel_names)))
            
        while True:
            smp_n, evt_n = ft.wait(next_smp, next_evt, 500)
            
            if smp_n < next_smp:
                print('Number of samples decreased -- getting new header.')
                break
        
            if evt_n > header.nEvents:
                evts = ft.getEvents([next_evt, evt_n-1])
                for e in evts:
                    print(e)
                next_evt = evt_n
                                        
            if next_smp == smp_n:
                continue #no new samples
              
            data = ft.getData([next_smp, smp_n-1])
            next_smp = smp_n
            
            if not triggered:
                trig = trigger.check(
                    MultiChannelData(data, 
                                     channel_names,
                                     sampling_frequency = header.fSample)
                )
                if trig == None:
                    if time() > time_start + timeout_sec:
                        raise IOError('Did not trigger within timeout')
                    continue #trigger not reached
                start_index = trig
                start_index += \
                    int(round(header.fSample * (trigger.start_time - 
                                                trigger.trigger_time)))
                triggered = True
            
            if start_index >= data.shape[0]:
                start_index -= data.shape[0]
                continue #triggered but acquisition delayed to next buffer
            
            #OK, triggering has occured and period starts in this buffer
            samples_to_copy = min(n_samples - buf_index, 
                                  data.shape[0] - start_index)
                                  
            ret_buffer[buf_index:buf_index + samples_to_copy,:] = \
                data[start_index:start_index + samples_to_copy,:]
            buf_index += samples_to_copy
            start_index = 0
            
            start_time = trigger.start_time if trigger != None else 0.0
            if buf_index == n_samples:
                return MultiChannelData(ret_buffer, channel_names, 
                                        start_time = start_time,
                                        sampling_frequency = header.fSample)
                
    

            
