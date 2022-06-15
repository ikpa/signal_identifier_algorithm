import random as ran
import numpy as np

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