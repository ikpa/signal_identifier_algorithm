# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:55:01 2013

A module for computing eddy currents in a thin conducting surface.

This is based on earlier Matlab code by Koos Zevenhoven 2010/2011.

@author: Koos Zevenhoven
"""
from __future__ import unicode_literals, division
import numpy as np
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import hashlib

def _to_hashable(obj):
    if isinstance(obj, tuple):
        return tuple(_to_hashable(o) for o in obj)
    elif isinstance(obj, np.ndarray):
        return _to_hashable((obj.shape, bytes(obj)))
    elif isinstance(obj, bytes):
        return memoryview(hashlib.sha1(obj).digest()[:8]).cast('L')[0]
    elif isinstance(obj, (int,
                          np.signedinteger,
                          np.unsignedinteger,
                          np.floating,
                          float,
                          np.complexfloating,
                          )):
        return _to_hashable(np.array(obj))
    elif isinstance(obj, str):
        return _to_hashable(obj.encode('utf8'))
    elif isinstance(obj, CoilSystem):
        return obj
    else:
        # don't know if this has a useful hash
        raise NotImplementedError("No hash for type: " + type(obj).__name__)


_lastflush = -float('inf')
_messages = []
def printprogress(what, level = 2):
    from . import verbose
    import time
    if verbose == 0:
        return
    if verbose >= level:
        _messages.append(what + '\n')
    else:
        import sys
        if len(_messages) == 0 or _messages[-1] != '.':
            _messages.append('.')
    global _lastflush
    if time.time() - _lastflush > 0.3:
        sys.stdout.write(''.join(reversed(_messages)))
        _lastflush = time.time()



def grid_in_space(center, h_and_v_vecs, h_and_v_points):
    """Return a point grid on a 2D rectangle in 3D space.

    Arguments:
        center -- center point of rectangle
        h_and_v_vecs = (hv, vv) -- 'horizontal' and 'vertical' sides
        h_and_v_points = (hp, vp) -- number of grid points in each direction

    Return:
        grid points in array of shape (3, v_points, h_points) NOTE ORDER!
        (horizontal_size, vertical_size)
        (horizontal_point_separation, vertical_point_separation)
        normal vector of the plate
    """
    h_vec, v_vec = h_and_v_vecs
    h_points, v_points = h_and_v_points
    lowerleft = center - h_vec/2 - v_vec/2
    ctmp = np.cross(h_vec, v_vec)
    normal = ctmp / np.sqrt(np.sum(ctmp*ctmp))

    h_step = h_vec / (h_points - 1)
    v_step = v_vec / (v_points - 1)
    h_size = np.sqrt(np.sum(h_vec**2))
    v_size = np.sqrt(np.sum(v_vec**2))
    h_delta = h_size / (h_points - 1)
    v_delta = v_size / (v_points - 1)

    grid = np.zeros((3, v_points, h_points))
    #X, Y, Z = grid

    for vi in range(v_points):
        for hi in range(h_points):
            grid[:,vi,hi] = lowerleft + hi*h_step + vi*v_step

    return grid, (h_size, v_size), (h_delta, v_delta), normal


class CoilSystem(object):
    """A surface (or surfaces) with an associated eddy-current basis.

    At least so far this is an abstract class.
    """
    #TODO turn this into a more general circuit system and derive it from
    # an even more general linear system
    def __hash__(self):
        raise NotImplementedError
        
    def _get_cache_file(self, var_name):
        from . import get_cache_dir
        cachedir = get_cache_dir()
        if not cachedir.is_dir():
            if cachedir.exists():
                raise FileExistsError("Cache dir %s exists but is not a directory"
                                      % cachedir)
            cachedir.mkdir()

        return cachedir / ('%015i-%s' % (hash(self) % 10**15, var_name))

    def _save_cache(self, var_name, obj):
        from pickle import dump
        with open(str(self._get_cache_file(var_name)), 'wb') as f:
            dump(obj, f)

    def _load_cache(self, var_name):
        from pickle import load
        file = self._get_cache_file(var_name)
        if file.is_file():
            with open(str(file), "rb") as f:
                obj = load(f)
        else:
            return None
        return obj

    def generated_field(self, points, state):
        """Compute the field from eddy currents of the given state.

        Arguments:
            points -- field points to calculate field at
            state -- amplitues of modes
        """
        if state.size != self.count_modes:
            #TODO see exception
            raise NotImplementedError("TODO: state = only the right elements")
        raise NotImplementedError("Fix the dot product")
        return state.dot(self.generated_fields(points))

    def __repr__(self):
        try:
            prefix = str(self.parent()) + "->"
        except:
            prefix = ""

        try:
            label = self.label
        except:
            label = "noname"

        return "{}{}".format(prefix, label)

    def coupling_vector_dipole(self, dipole_p, dipole):
        """Get a vector of mutual inductances from a dipole.

        This is "the flux through each basis function" from the dipole.
        """
        fields = self.generated_fields(dipole_p)
        #dipole = dipole.reshape([3] + [1] * (len(dipole.shape) - 1))
        return np.tensordot(fields,dipole, axes = [[0],[0]])

    @property
    def M(self):
        from . import _use_disk_cache
        if hasattr(self, '_M') and hasattr(self, '_hash_at_cache'):
            if self._hash_at_cache == hash(self):
                return self._M
        if _use_disk_cache:
            obj = self._load_cache('M')
            if obj is None:
                obj = self.mutual_inductances(self)
            self._M = obj
            self._hash_at_cache = hash(self)
            self._save_cache('M', self._M)
            return self._M

    @property
    def R(self):
        try:
            return self._R
        except:
            self._R = np.eye(self.count_modes()) / \
                (self.conductivity * self.thickness)
            return self._R

class TransformedSystem(CoilSystem):
    """An interface to an inductive circuit system with change of basis."""

    def __init__(self, system, transformation_matrix):
        """Wrap the given system, transformed by the given matrix."""
        self.system = system
        self.transformation = transformation_matrix

    #TODO everything


class CoilSystemList(CoilSystem):
    """A set of (sub-)surfaces with an associated eddy-current basis."""

    def __init__(self, subsurfaces, label = None):
        """Combine the given subsurfaces in an CoilSystemList."""
        if not label:
            label = "CoilSystemList"
        self.label = label

        self.surfaces = list(subsurfaces)
        from weakref import ref
        class statics:
            next_index = 0
        def handle_children():
            statics.next_index = 0
            for s in self.surfaces:
                s.parent = ref(self)
                yield statics.next_index
                statics.next_index += s.count_modes()

        self.first_indices = list(handle_children())
        self.n_modes = statics.next_index

    def __hash__(self):
        return hash(_to_hashable((self.label,) + tuple(self.surfaces)))

    def count_modes(self):
        """Count the number of modes in this surface."""
        #return sum(s.count_modes() for s in self.surfaces)
        return self.n_modes

    def mutual_inductances(self, other, M = None):
        """Make mutual inductance matrix for a list of separate surfaces.

        Arguments:
            other -- other surface with a basis / bases

        Returns:
            mutual inductance matrix according to matrix indices.
        """
        if M is None:
            M = np.empty((self.count_modes(), other.count_modes()))

        for s, i in zip(self.surfaces, self.first_indices):
            n = s.count_modes()
            s.mutual_inductances(other, M[i:(i + n), :])

        if other == self:
            M += M.transpose()
            M *= 0.5
            self._M = M
            #note that each mutual has been calculated twice
            #take average of each two corresponding values (force hermitianity)

        return M

    def generated_fields(self, points):
        """Compute the fields generated by the basis in this surface.

        Unit current amplitudes are assumed in the basis functions.

        Arguments:
            points -- 3-vector field points, shape (3, *point_dimensions)

        Returns:
            numpy array of shape (3, n_basis[, *point_dimensions])
        """
        return np.concatenate([s.generated_fields(points)
                                  for s in self.surfaces], axis=1)

    def plot_geometry(self, ax = None):
        """Make a 3D plot of the system geometry."""
        if ax is None:
            from mpl_toolkits.mplot3d import Axes3D
            fignum = plt.get_current_fig_manager().num
            fig = plt.figure(fignum)
            ax = Axes3D(fig)

        for s in self.surfaces:
            s.plot_geometry(ax)


        return ax


class EddyPlate(CoilSystem):
    """A plate with a 2-D sine-series-based eddy-current basis."""

    MIN_POINTS_PER_LOBE = 7
    MIN_POINTS = 15

    def __init__(self, center, h_and_v_vecs, h_and_v_max_orders=(None,None),
                 detail_scale=(None,None), first_index=0, label=None):
        """Create an eddy current basis for a rectangular plate in space.

        Arguments:
            center -- center point of plate
            h_vec, v_vec -- 'horizontal' and 'vertical' side vectors
            h_order, v_order -- max basis function order horizontally
                and vertically
            sampling_scale -- scale of computation detail (one or two numbers)
            first_index -- index at which to start numbering basis functions
        """
        if not label:
            label = "EddyPlate"
        self.label = label

        h_vec, v_vec = h_and_v_vecs
        detail_scale = np.array(np.broadcast_to(detail_scale, (2,)))
        detail_res = np.empty(detail_scale.shape, dtype='i')
        for i in range(2):
            if detail_scale[i] is None:
                if h_and_v_max_orders[i] is None:
                    raise ValueError("Either order or detail scale needed")
                detail_scale[i] = float('inf')
            detail_res[i] = int(round(np.sqrt(np.sum(np.square((h_vec,v_vec)[i])))
                                    / detail_scale[i]))


        h_order, v_order = [min(o for o in ords if o)
                                for ords in zip(h_and_v_max_orders, detail_res)]

        hres = max(r for r in (h_order * self.MIN_POINTS_PER_LOBE,
                               detail_res[0],
                               self.MIN_POINTS) if r)
        vres = max(r for r in (v_order * self.MIN_POINTS_PER_LOBE,
                               detail_res[1],
                               self.MIN_POINTS) if r)

        grid, (h_size, v_size), (h_delta, v_delta), normal = \
            grid_in_space(center, (h_vec, v_vec), (hres, vres))


        midpoints = 0.25*(grid[:,:-1,:-1] + grid[:,:-1,1:] +
                          grid[:,1:,:-1] + grid[:,1:,1:])

        (Xb,Yb) = np.meshgrid(np.linspace(0, h_size, hres),
                              np.linspace(0, v_size, vres))

        Xbj = 0.25*(Xb[:-1,:-1] + Xb[:-1,1:] + Xb[1:,:-1] + Xb[1:,1:])
        Ybj = 0.25*(Yb[:-1,:-1] + Yb[:-1,1:] + Yb[1:,:-1] + Yb[1:,1:])

        eh = h_vec / h_size
        ev = v_vec / v_size

        basis = []
        basis_nabla = []
        basis_order = []
        for vi in range(1, v_order + 1):
            for hi in range(1, h_order + 1):
                A = 2/np.sqrt((v_size*h_size*
                               ((hi/h_size)**2 + (vi/v_size)**2)))/np.pi

                hppw = hi * np.pi / h_size
                vpph = vi * np.pi / v_size

                psi = A*np.sin(Xb*hppw)*np.sin(Yb*vpph)

                xcos_ysin = np.cos(Xbj*hppw) * np.sin(Ybj*vpph)
                xsin_ycos = np.sin(Xbj*hppw) * np.cos(Ybj*vpph)

                nabla_psi = (A * hppw * eh.reshape((3,1,1)) * xcos_ysin +
                             A * vpph * ev.reshape((3,1,1)) * xsin_ycos)

                basis.append(psi.reshape((1,) + psi.shape))
                basis_nabla.append(nabla_psi.reshape((1,) + nabla_psi.shape))
                basis_order.append(np.array([[hi,vi]]));

        self.size = (h_size, v_size)
        self.deltas = (h_delta, v_delta)
        self.normal = normal
        self.center = center
        self.grid_psi = grid
        self.psi = np.concatenate(basis)
        self.grid_nabla_psi = midpoints
        self.nabla_psi = np.concatenate(basis_nabla)
        self.n_modes = h_order*v_order
        self.mode_order = np.concatenate(basis_order)

        #make mask to fix integral terms at edges
        mask = 0.5*np.ones((self.grid_psi.shape[-2:]))
        mask[1:-1,1:-1] = np.ones((mask.shape[0]-2, mask.shape[1]-2))
        mask[0,0] = 0.25
        mask[-1,-1] = 0.25
        mask[0,-1] = 0.25
        mask[-1,0] = 0.25
        #TODO use: scipy.integrate.simps(y, x=None, dx=1, axis=-1, even='avg')[source]
        self._integral_mask = mask

    def __hash__(self):
        data = (self.label, self.size, self.deltas, self.normal, self.center,
                self.grid_psi, self.psi, self.grid_nabla_psi, self.nabla_psi,
                self.n_modes, self.mode_order)
        return hash(_to_hashable(data))


    def count_modes(self):
        return self.n_modes

    def integral(self, values):
        """Calculate integral of a function sampled on the plate.

        The grid (grid_psi or grid_nabla_psi) is selected depending on
        the size of the argument array.

        Arguments:
            values -- array of values on the grid
        """
        #TODO use: scipy.integrate.simps(y, x=None, dx=1, axis=-1, even='avg')[source]
        if values.shape[-2:] == self.grid_psi.shape[-2:]:
            mask = self._integral_mask
            if len(values.shape) == 3: #vector data, prepare for broadcast
                mask = mask.reshape((1,) + mask.shape)
        elif values.shape[-2:] == self.grid_nabla_psi.shape[-2:]:
            mask = 1.0 # no mask needed since on center points
        else:
            raise Exception("Wrong array shape to integrate over")

        dS = np.prod(self.deltas)
        return dS*(mask * np.array(values)).sum()

    def generated_fields_(self, point):
        """Compute the fields generated by the basis functions in this plate.

        NOTE THIS IS NOT COMPATIBLE ANYMORE: LEFT IT IN FOR CHECKING

        Unit current amplitudes are assumed in the modes.

        Arguments:
            point -- a 3-vector position of the field point

        Returns:
            numpy array of shape (modes, xyz)
        """
        #TODO can all of this be vectorized for multiple points?
        dR = point.reshape((3,1,1)) - self.grid_nabla_psi
        dR2 = (dR**2).sum(axis=0)
        dR3 = np.sqrt(dR2)*dR2
        dRm3 = 1/dR3
        n = self.normal

        tmp1 = (n.reshape((3,1,1)) * dR).sum(axis=0) * dRm3
        tmp1 = tmp1.reshape((1,) + tmp1.shape)

        dS = np.prod(list(self.deltas))
        k = 1e-7 #\mu_0/4pi (exact in SI)

        def sum_tail_dims(arr, n_dims):
            tmp = arr.reshape(arr.shape[:-n_dims] + (-1,))
            return tmp.sum(axis = len(tmp.shape)-1)

        #I1 is a vector per mode, I2 a scalar per mode
        I1 = sum_tail_dims((tmp1*self.nabla_psi),2)
        I2 = sum_tail_dims((dR*dRm3*self.nabla_psi),3)

        return dS*k*(I2.reshape(I2.shape + (1,))*n-I1)

    def generated_fields(self, points):
        """Compute the fields generated by the basis functions in this plate.

        Unit current amplitudes are assumed in the basis functions.

        Arguments:
            points -- a number of 3-vector points to calculate field at:
                shape = (3, n_points[, npoints2, ...])

        Returns:
            numpy array of shape (3, n_basis, n_points[, n_points2, ...])
        """
        from .vectorizetools import sum_keep_axis, insert_dummy_axis

        dS = np.prod(self.deltas)
        k = 1e-7 #\mu_0/4pi (exact in SI)
        n = self.normal

        #TODO: save memory by limiting the maximum chunk of field points
        #used at once and loop over whole buffer

        #The axes could be arranged to reduce swapping axes and reshaping,
        #but this should not be a bottleneck here

        point_dims = points.shape[1:]
        points = points.reshape((3, -1)).swapaxes(0,1).copy()

        dR = points.reshape((-1,3,1,1)) - self.grid_nabla_psi
        dR_dRm3 = dR * sum_keep_axis(dR**2, axis=1)**(-3./2.)
        dR_dRm3 = insert_dummy_axis(dR_dRm3, 1)
        del dR

        #Integrals: I1 is a vector per basis func, I2 a scalar per basis func
        I2 = np.einsum("...ijk,...ijk", dR_dRm3, self.nabla_psi)

        n_dot_dR_dRm3 = sum_keep_axis(n.reshape((1,1,3,1,1)) * dR_dRm3,
                                      axis=2)
        del dR_dRm3
        I1 = np.einsum("...ij,...ij", n_dot_dR_dRm3, self.nabla_psi)
        del n_dot_dR_dRm3

        ret = dS*k*(insert_dummy_axis(I2, None) * n - I1).swapaxes(0,2)
        #swapped axes like this: (points, basis, 3)->(3, basis, points)

        n_points = np.prod(point_dims)
        printprogress("generate_fields(): {} *for* {} points".format(self, n_points))
        ret = ret.reshape(ret.shape[:2] + point_dims).copy()
        return ret

    def mutual_inductances(self, other, mutuals = None):
        """Calculate mutual inductances of modes into inductance matrix.

        Calculate mutual inductances of self with other. Optionally,
        a readily allocated array or sub-array view can be given to build
        the mutual inductance matrix into.
        """

        if mutuals is None:
            mutuals = np.empty((self.count_modes(), other.count_modes()))

        points = self.grid_psi
        n = self.normal

        if False: # non-vectorized version
            normalfield = np.empty((other.n_modes,) + points.shape[-2:])

            for vi in range(points.shape[-2]):
                for hi in range(points.shape[-1]):
                    #TODO can generated_fields be vectorized for multiple points?
                    field = other.generated_fields(points[:,vi,hi])
                    normalfield[:,vi,hi] = \
                        (n.reshape((1,3))*field).sum(axis=1)
        else: # vectorized version
            fields = other.generated_fields(points)
            normalfield = (n.reshape((3,1,1,1)) * fields).sum(axis = 0)

        for i1 in range(self.n_modes):
            for i2 in range(other.n_modes):
                m = self.integral(self.psi[i1,:,:] * normalfield[i2,:,:])
                mutuals[i1, i2] = m
        printprogress("mutual_inductances(): {} *with* {}".format(self, other))
        return mutuals

    def plot_geometry(self, ax = None):
        """Make a 3D plot of the system geometry."""
        if ax is None:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fignum = plt.get_current_fig_manager().num
            fig = plt.figure(fignum)
            ax = Axes3D(fig)

        import random
        c = ('r','g', 'b', 'c', 'm', 'y')[random.randint(0,5)]

        ax.scatter(*self.grid_psi[:,::4,::4].reshape(3,-1), label=self.label, c=c)
        return ax

#if __name__ == "__main__":
#    b = EddyPlate(np.array([0,0,0]),(np.array([1,0,0]),np.array([0,1,0])), (4,3))
