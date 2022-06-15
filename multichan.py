# -*- coding: utf-8 -*-
"""
Copyright (C) Koos C. J. Zevenhoven. All rights reserved.

Copying, modifying or distributing this library or sourcecode, 
via any medium, without permission from the author, is strictly 
prohibited.

This file was created on Tue Aug 13 22:40:22 2013

Tools for working with multi-channel data / sequences / arrays.

@author: Koos Zevenhoven
"""

from collections import OrderedDict

def wildcardkeys(all_keys, key_list):
    """Return matching strings, allowing lists and wildcards.
    
    Arguments:
        all_keys -- list of all allowed keys
        key_list -- a key or list of keys to match, 
            can contain * and ? as wildcards
    
    Returns: 
        iterable of matching keys
        
    NOTE: 
        Currently the output order may not be the same as in the original 
        key list!
    """
            
    import fnmatch
    
    if isinstance(key_list, str):
        key_list = [key_list]
    
    assert all(isinstance(k, str) for k in key_list)

    for key in key_list:
        for f in sorted(fnmatch.filter(all_keys, key)):
            yield f


class ChannelPool(OrderedDict):
    """A base class for multi-channel stuff with named channels."""
    
    #TODO a function that determines the best subclass for returnvalues
    # of each and foreach
    

    def __repr__(self):
        return "<{} object: {} channels; first one of type {}>" \
            .format(type(self).__name__, len(self), 
                    type(next(iter(self.values()))).__name__)
    
    def match(self, chan_list):
        """Return an iterable of channel names based on list (wildcards ok)."""
        return wildcardkeys(self, chan_list)
    
    def get_channels(self, chan_list):
        """Return an iterable of channels by names (wildcards ok)."""
        for ch in self.match(chan_list):
            yield self[ch]
            
    def subpool(self, chan_list):
        """Return a view to self, with picked channels (wildcards ok)."""
        
        class SubPool(type(self)):
            def __init__(self, superpool, chans):
                for ch in chans:
                    self[ch] = superpool[ch]
                
        sub = SubPool(self, self.match(chan_list))
        
        return sub
    
    def foreach(self, func, return_type = None):
        """Return a pool of results of func applied to all values.
        
        To call a method or get an attribute from all values, there is some
        syntactic sugar using the "each" property instead:
            result_pool = pool.each.do_something()
            
        Or:
            attribute_pool = pool.each.some_attribute
        
        Setting with "each" is also possible:
        
            pool.each.some_attribute = False
        
        
        Arguments:
            func: function to be applied to the values.
            return_type: dict-like class for the result (None -> ChannelPool)
        """
        
        if return_type == None:
            return_type = ChannelPool
        return return_type(zip(iter(self.keys()), 
                                map(func, iter(self.values()))))
    
    class Each(object):
        """ A proxy representing each value in a dict-like object. """
        __slots__ = ["_obj", "__weakref__"]
        def __init__(self, obj):
            """Make an "each" proxy for the dict-like obj. """
            object.__setattr__(self, "_obj", obj)
        

        # Proxy that forwards actions to each channel/value

        def __getattribute__(self, name):
            obj = object.__getattribute__(self, "_obj")
            ret = obj.foreach(lambda x: getattr(x, name))
            
            try:
                if hasattr(next(iter(ret.values())), "__call__"):
                    # The property is callable => FunctionPool
                    return CallablePool(ret)
            except StopIteration:
                pass
            return ret
                    
        def __delattr__(self, name):
            what = lambda x: delattr(x, name)
            return object.__getattribute__(self, "_obj").foreach(what)
            
        def __setattr__(self, name, value):
            what = lambda x: setattr(x,  name, value)
            return object.__getattribute__(self, "_obj").foreach(what)
            
        def __bool__(self):
            what = lambda x: bool(x)
            return object.__getattribute__(self, "_obj").foreach(what)
            
        def __str__(self):
            what = lambda x: str(x)
            return object.__getattribute__(self, "_obj").foreach(what)
            
        def __repr__(self):
            what = lambda x: repr(x)
            return object.__getattribute__(self, "_obj").foreach(what)
            
        def __getitem__(self, *args, **kwargs):
            what = lambda x: x.__getitem__(*args, **kwargs)
            return object.__getattribute__(self, "_obj").foreach(what)
            
        def __setitem__(self, *args, **kwargs):
            what = lambda x: x.__setitem__(*args, **kwargs)
            object.__getattribute__(self, "_obj").foreach(what)
            
        #TODO add more functionality like __call__?
    
    @property
    def each(self):
        return self.Each(self)
        

class CallablePool(ChannelPool):
    """A pool with a callable for each channel.
    
    Calling the pool applies the call to all values and returns the results
    in the type given by the attribute return_type (default: ChannelPool).
    """
    return_type = ChannelPool
    
    def __call__(self, *args, **kwargs):
        """Call the function associated to each object.
        
        Arguments:
            args, kwargs: Arguments passed on to each callable
        Return:
            dict-like object of type self.return_type,
            with collected return values
        """
        return self.foreach(lambda x: x(*args, **kwargs), 
                            return_type = self.return_type)
        

class PoolExpression(ChannelPool):
    """A ChannelPool that can be a (channel-wise) sum or product of others.
    
    The channels/values of the pool must support the same operations as used
    on the pool(s). 
    """
    
    pass

class ExpressionMeta(type):
    """A metaclass for making classes that allow building expressions."""
    pass
    
