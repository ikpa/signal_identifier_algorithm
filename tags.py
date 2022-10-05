# -*- coding: utf-8 -*-
"""
Copyright (C) Koos C. J. Zevenhoven. All rights reserved.

Copying, modifying or distributing this library or sourcecode, 
via any medium, without permission from the author, is strictly 
prohibited.

This file was created on Wed May 29 17:56:45 2013

A Tag module for use with PyFunk. A tag is a parameter value (often an instant
of time) that has an associated name. Tags can also be defined relative to
other tags.

@author: Koos Zevenhoven
"""

from __future__ import unicode_literals, division, print_function
from functools import reduce


_noname_number = 0
def ensure_tag(value, name = None):
    """Return a tag with the given value, or value itself if already a tag.
    
    If a new tag is created, it will have the given name unless it is None.
    name == None, a name such as noname21 will be given.
    """
    global _noname_number
    if isinstance(value, Tag):
        return value
    
    if name == None:
        name = "noname" + str(_noname_number)
        _noname_number += 1
    
    #return a new tag
    return Tag(name, value)

class Tag(object):
    """Gives a lable to a parameter value or instant in time.
    
    Instances of this class or its subclasses can be used to tag instants
    of time or to wrap a real-valued parameter. In general, Tag objects
    are mutable and their value can depend on other objects.
    """
    
    def __init__(self, name, value):
        """Create tag with name at value/instant given by value.
        
        If value is a Tag, an alias with the given name is created"""
        self.name = name
        self.set_value(value)
    
    @property
    def value(self):
        if isinstance(self._value, Tag):
            return self._value.value
        else:
            return self._value
    
    def set_value(self, value):
        self._value = value
        
    def __float__(self):
        return float(self.value)
        
    def __int__(self):
        return int(self.value)
        
    def __repr__(self):
        return self.name + ": " + str(float(self.value))
        
    def toIndependent(self, name = None):
        """Make an independent copy of the current value.
        
        The returned tag will be independent of other tags. If a name
        other than None is given, the copy will get the given name. Otherwise
        the name will be copied also.
        """
        if name == None:
            name = self.name
        return Tag(name, self.value)

    def _get_dep_candidates(self):
        """Return an iterable of possible direct dependencies.
        
        The elements in the iterable are all either direct dependencies of this
        tag or not instances of Tag. 
        
        See get_direct_dependencies. This is to be overrided in subclasses.
        """
        return []
        
                                 
    def get_direct_dependencies(self):
        """The direct dependencies on other Tags (not examined recursively).
        
        Overriding this (or _get_dep_candidates()) appropriately in subclasses 
        will make get_all_dependencies() work automatically.
        """
        return set(d for d in self._get_dep_candidates() if isinstance(d, Tag))

    def get_all_dependencies(self, who_asks = None):
        """Get a set of all (also indirect) dependencies of this tag."""
        if who_asks == self:
            raise RuntimeError("Tag depends on itsel.")
        elif who_asks == None:
            who_asks = self
            
        return reduce(lambda a, b: a|b, 
                      [d.get_all_dependencies(who_asks) 
                           for d in self.get_direct_dependencies()])

class ImmutableTag(Tag):
    """A simple constant tag whose value cannot be changed from anywhere."""
    def __init__(self, name, value):
        if isinstance(value, Tag):
            value = float(value)
        self._name = name
        self._value = value
        
    @property
    def value(self):
        return self._value
    
    @property
    def name(self):
        return self._name
    
    def set_value(self):
        raise RuntimeError("Cannot set the value of an immutable tag")

class SumTag(Tag):
    """A tag at one tag plus an offset."""
    
    def __init__(self, name, basetag, offset):
        """Create tag with name at basetag + offset.
        
        The offset can be a builtin type or a Tag, allowing 
        external modification of the delay time. The offset can be positive 
        (later/more than basetag) or negative (less/earlier).
        """
        self.name = name
        self._basetag = basetag
        self._offset = offset

    @property
    def value(self):
        return float(self._basetag) + float(self._offset)
        
    def _get_dep_candidates(self):
        return (self._basetag, self._offset)

class DiffTag(Tag):
    """A tag representing a difference between two values/instants."""
    
    def __init__(self, name, a, b):
        """Create tag with name at b - a
        
        The tags can be builtin numbers or a Tags, allowing 
        external modification. Note the order of subtraction, which is
        motivated by thinking of the length of an interval from a to b! 
        """
        self.name = name
        self._a = a
        self._b = b
    
    @property
    def value(self):
        return float(self._b) - float(self._a)
        
    def _get_dep_candidates(self):
        return (self._a, self._b)
        
class MulTag(Tag):
    """A tag at one tag times a factor value/tag."""
    
    def __init__(self, name, basetag, factor):
        """Create tag with name at basetag * factor.
        
        The factor can be a builtin number or a Tag, allowing 
        external modification. 
        """
        self.name = name
        self._basetag = basetag
        self._factor = factor
        
    @property
    def value(self):
        return float(self._basetag) * float(self._factor)
    
    def _get_dep_candidates(self):
        return (self._basetag, self._factor)


    
class Interval(Tag):
    """Defines one tag with respect to another based on interval value.
    
    This is an "active" difference tag, i.e., the start or end point will be
    defined by the value of this tag (which can be changed directly or given by
    another mutable tag).
    """

    def __init__(self, name, basetag, value, base_is_start = True):
        """Create an interval with basetag as the start(end) point.
        
        If is_start, basetag will be the starting point; otherwise the
        end point. The remaining end (start) point will be created as a
        relative tag depending on this Interval tag and basetag.
        """
        
        if not isinstance(basetag, Tag):
            #assume it is then a builting number type -> make tag
            basetag = Tag("BASE-" + name, basetag)
       
        self._base_is_a = base_is_start
        
        if self._base_is_a:
            self._a = basetag
            self._b = SumTag("END-" + name, self._a, self)
        else:
            self._b = basetag
            self._a = DiffTag("START-" + name, self, self._b)

        self.name = name
        self.value = value

    def get_start(self):
        """Get the start tag."""
        return self._a
        
    def get_end(self):
        """Get the end tag."""
        return self._b
        
    def get_direct_dependencies(self):
        return set([self._a if self._base_is_a else self._b])

# NOW DEFINE SOME USEFUL CONSTANTS

#: Plus infinity tag (instance of ImmutableTag).
PLUS_INF = ImmutableTag("PLUS_INFINITY", float("+inf"))

#: Minus infinity tag (instance of ImmutableTag).    
MINUS_INF = ImmutableTag("MINUS_INFINITY", float("-inf"))

#: Zero tag (instance of ImmutableTag)
ZERO = ImmutableTag("ZERO", 0.0)
