import random as ran
import numpy as np
from numpy.lib import recfunctions as rf
import pyfunk as pf
from coils.eddycurrents0 import diagonalize, transient_amplitudes, SplitBoxRoom, UNIT
from coils.coils import ScalarCoilPlate, DipoleSet
import matplotlib.pyplot as plt
import multiprocessing as mp

errtypes = ["saturation", "jump"]

a0 = 10
tau0 = 10
const0 = 10**3
b0 = 1.5
#wmin = 20
#wmax = 70
w0 = .1
phi0 = 2 * np.pi
sig_size = 500

ranerr_sigma = .05


def inv_step_response(system, coil, field_p, field_dir, length, mode_selection=None):
    """Get inverse step response of an eddy-current system to pulsed coil.

    Arguments:
        system: the eddy current system (CoilSystem object)
        coil: an CoilSystem object representing the coil
        field_p: the point at which the field is measured
        field_dir: a unit vector in the direction of the field measurement
        length: length of the inverse step response in seconds
    """

    def mode_responses():
        from functools import partial
        fun = lambda t, tau: np.exp(-t / tau)
        for lambda_, tau in zip(system.lambdas, system.tau):
            yield pf.Function(partial(fun, tau=tau), (0.0, length))

    basis = FunctionBasis(mode_responses())

    coil_c = coil.mutual_inductances(system).flatten()
    field_c = system.generated_fields(field_p).T.dot(field_dir.flatten())

    mode_input = system.V.T.dot(coil_c) / system.lambdas
    mode_output = system.V.T.dot(field_c)

    return basis.get_waveform(mode_input * mode_output)


def calc_response(pulse, resp_function):
    """Get system response to pulse, based on given inverse step response."""
    return -pulse.differentiate().convolve(resp_function)


class LinearCombination(pf.Waveform):
    """A waveform that is a linear combination of other waveforms."""

    def __init__(self, functions, coefficients):
        self.coefficients = coefficients
        self.functions = functions
        self.bound_tags = \
            tuple((max((b.bound_tags[0] for b in self.functions), key=lambda x: x.value),
                   min((b.bound_tags[1] for b in self.functions), key=lambda x: x.value)))

    def sample(self, positions):
        return np.sum([bf.sample(positions) * c
                       for bf, c in zip(self.functions, self.coefficients)], axis=0)


class FunctionBasis(object):
    """A waveform function basis"""

    def __init__(self, basis_waveforms):
        self.functions = list(basis_waveforms)

    def get_waveform(self, coefficients):
        """Get the waveform based on a coefficient or 'coordinate' vector."""
        return LinearCombination(self.functions, coefficients)

def simple_one_flat(x, n):
    jump_points = [int(n * ran.random()), int(n * ran.random())]
    jump_points.sort()
    #print(jump_points)
    x1 = jump_points[0]
    x2 = jump_points[1]
    y = 3 * 10**(-8) * ran.random()
    y1 = 3 * 10**(-8) * ran.random()
    y2 = 3 * 10**(-8) * ran.random()

    y_arr = []

    for x_val in x:
        if x_val >= x1 and x_val <= x2:
            slope = 0
            b = y

        if x_val < x1:
            slope = (y1 / x1)
            b = y - slope * x1

        if x_val > x2:
            slope = (y2 - y) / (n - x2)
            b = y - slope * x2

        val = slope * x_val + b
        y_arr.append(val)

    return y_arr

def simple_many_flat(x, n, n_flats, flats_same=False):
    jump_points = [0, n]
    flat_ys = [3 * 10 ** (-8) * ran.random(), 3 * 10 ** (-8) * ran.random()]

    if flats_same:
        y_val = 3 * 10 ** (-8) * ran.random()

    for i in range(n_flats):
        jump_points.append(int(n * ran.random()))
        jump_points.append(int(n * ran.random()))

        if not flats_same:
            y_val = 3 * 10 ** (-8) * ran.random()

        flat_ys.append(y_val)

    jump_points.sort()

    y_arr = []
    for x_val in x:

        for i in range(n_flats):
            j = i * 2
            x1 = jump_points[j - 1]
            x2 = jump_points[j]
            x3 = jump_points[j + 1]
            x4 = jump_points[j + 2]

            y_prev = flat_ys[i]
            y_current = flat_ys[i + 1]
            y_next = flat_ys[i + 2]

            if x_val > x2 and x_val < x3:
                y_arr.append(y_current)
                break
            elif x_val > x1 and x_val < x2:
                slope = (y_current - y_prev) / (x2 - x1)
                b = y_current - slope * x2
            elif x_val > x3 and x_val < x4:
                slope = (y_next - y_current) / (x4 - x3)
                b = y_current - slope * x3
            else:
                continue

            val = slope * x_val + b
            y_arr.append(val)
            break

    return y_arr

def signal_func(t, a, tau, const, b, w, phi):
    return a * np.exp(-t/tau) + b * np.sin(w * t + phi) + const

def saturation_sig(t_space, start_index, a, tau, const, b, w, phi, containsNoise=True):
    #noise1 = np.random.normal(0, ranerr_sigma, start_index)
    sig1 = signal_func(t_space[:start_index], a, tau, const, b, w, phi)
    #noise2 = np.random.normal(0, ranerr_sigma, sig_size - start_index)
    sig2 = np.full(sig_size - start_index, sig1[start_index - 1])

    if containsNoise:
        noise = np.random.normal(0, ranerr_sigma, sig_size)
    else:
        noise = np.zeros(sig_size)

    signal = np.append(sig1, sig2) + noise
    return signal

def jump_sig(t_space, start_index, a, tau, const, b, w, phi, containsNoise=True):
    #noise1 = np.random.normal(0, ranerr_sigma, start_index)
    sig1 = signal_func(t_space[:start_index], a, tau, const, b, w, phi)
    errconst = 1.5 * ran.random()
    #noise2 = np.random.normal(0, ranerr_sigma, sig_size - start_index)
    sig2 = signal_func(t_space[start_index:], a, tau, const, b, w, phi) + errconst

    if containsNoise:
        noise = np.random.normal(0, ranerr_sigma, sig_size)
    else:
        noise = np.zeros(sig_size)

    signal = np.append(sig1, sig2) + noise
    return signal

def normal_sig(t_space,a, tau, const, b, w, phi, containsNoise=True):
    if containsNoise:
        noise = np.random.normal(0, ranerr_sigma, sig_size)
    else:
        noise = np.zeros(sig_size)

    signal = signal_func(t_space, a, tau, const, b, w, phi) + noise
    return signal

def gen_signal(containsError = False, containsNoise=True):
    a = a0 * ran.random()
    tau = tau0 * ran.random()
    const = const0 * ran.random()
    b = b0 * ran.random()
    #w = ran.uniform(wmin, wmax)
    w = w0 * ran.random()
    phi = phi0 * ran.random()

    print("a", a)
    print("tau", tau)
    print("const", const)
    print("b", b)
    print("w", w)
    print("phi", phi)

    t_space = np.linspace(0, 50, sig_size)

    if containsError:
        err_type = ran.choice(errtypes)
        print("error type", err_type)
        start_index = ran.randrange(int(sig_size * 0.05), int(sig_size * 0.3))
        print("error start time", t_space[start_index])

        if err_type == "saturation":
            signal = saturation_sig(t_space, start_index, a, tau, const, b, w, phi, containsNoise)

        if err_type == "jump":
            signal = jump_sig(t_space, start_index, a, tau, const, b, w, phi, containsNoise)

    else:
        signal = normal_sig(t_space,a, tau, const, b, w, phi, containsNoise)
        start_index = None


    return signal, t_space, start_index

def prepare_args(detectors, names, isr_length, room, coil, pulse):
    n_detecs = len(detectors)
    #args = np.empty((n_detecs, 7))
    args = []

    for i in range(n_detecs):
        args.append([detectors[i], names[i], isr_length, room, coil, pulse])

    #print(args)

    return args

def parallel_eddy_func(args):
    detector = args[0]
    r = detector[:3, 3]
    v = detector[:3, 2]
    name = args[1]
    isr_length = args[2]
    room = args[3]
    coil = args[4]
    pulse = args[5]

    print("calculating isr for " + name)

    isr = inv_step_response(room, coil,
                              r, v, isr_length).to_samples()

    print("calculating response")

    response = calc_response(pulse, isr)

    return response.to_samples()._samples

def polarizing_coil():
    p_n = 240.0
    p_r = 0.19

    p_center = np.array([0, 0, -3.0 * 0.0254])  # position of coil center
    p_dipole = np.array([0, 0, 1.]) * p_n * np.pi * p_r ** 2  # dipole moment
    p_coil = DipoleSet(p_center, p_dipole)
    return p_coil

def dyna_coil():
    c_side = 1.85
    c_turns = 30
    c_res = 100
    c_center = -4 * UNIT.FOOT + 1.125
    c_coil = ScalarCoilPlate(c_center,
                             (np.array([c_side, 0, 0]), np.array([0, c_side, 0])),
                             np.ones((1, c_res, c_res)) * c_turns,
                             label="dynacan_coil")
    return c_coil

def simulate_eddy_parallel(detectors, names):
    room = SplitBoxRoom(detail_scale=UNIT.FOOT,
                        thickness=10e-3)
    diagonalize(room)

    coil = polarizing_coil()

    I = 200.0
    ramp_time = 11.6e-3
    pulse = I * pf.RampedBox(1.0,
                                 (-0.2, -ramp_time),
                                 (10e-3, ramp_time),
                                 (pf.LinearRamp, pf.QuarterCosineRamp))

    isr_length = 500e-3

    args = prepare_args(detectors, names, isr_length, room, coil, pulse)
    cpus = mp.cpu_count()

    pool = mp.Pool(cpus)
    responses = pool.map(parallel_eddy_func, args)

    return responses


def simulate_eddy(detectors):
    # Some ft units are used to define this, because it is how the Berkeley
    # shielded room was built, but everything gets quickly converted to SI units
    room = SplitBoxRoom(detail_scale=UNIT.FOOT,
                        thickness=10e-3)
    diagonalize(room)

    # polarizing coil (modeled as a dipole)
    # 2022-06: tämän suuntaa on myös helppo kääntää t. Koos

    coil = polarizing_coil()

    isr_length = 500e-3
    p_ramp_time = 11.6e-3
    I = 200.0
    p_pulse = I * pf.RampedBox(1.0,
                                 (-0.2, -p_ramp_time),
                                 (10e-3, p_ramp_time),
                                 (pf.LinearRamp, pf.QuarterCosineRamp))

    responses = []
    for name in detectors:
        detector = detectors[name]
        r = detector[:3, 3]
        v = detector[:3, 2]

        print("calculating isr for " + name)

        isr = inv_step_response(room, coil,
                                  r, v, isr_length).to_samples()
        print("calculating response")

        p_trans = calc_response(p_pulse, isr)
        responses.append(p_trans.to_samples()._samples)

    return responses

