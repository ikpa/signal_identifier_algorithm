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

def simple_one_flat(x, n):
    jump_points = [int(n * ran.random()), int(n * ran.random())]
    jump_points.sort()
    print(jump_points)
    x1 = jump_points[0]
    x2 = jump_points[1]
    y = 3 * 10**(-8) * ran.random()
    y1 = 3 * 10**(-8) * ran.random()
    y2 = 3 * 10**(-8) * ran.random()

    y_arr = []

    for x_val in x:
        if x_val > x1 and x_val < x2:
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

def simple_many_flat(x, n, n_flats):
    jump_points = [0, n]
    flat_ys = [3 * 10 ** (-8) * ran.random(), 3 * 10 ** (-8) * ran.random()]

    for i in range(n_flats):
        jump_points.append(int(n * ran.random()))
        jump_points.append(int(n * ran.random()))

        flat_ys.append(3 * 10 ** (-8) * ran.random())

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