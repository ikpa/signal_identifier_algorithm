import math

import numpy as np
import time
import helper_funcs as hf
from operator import itemgetter


"""all sensitivity and weight values in this file have been determined
experimentally and changing them will affect the accuracy of the program
significantly"""

sample_freq = 10000 # sampling frequency of the squids

def filter_start(signal, offset=50, max_rel=0.05, debug=False):
    """filter the jump in the beginning of the signal. returns the index
    where the jump has ended."""
    max_i = int(max_rel * len(signal))
    grad = np.gradient(signal)
    new_grad = grad[:max_i]
    new_grad = abs(np.array(new_grad))
    largest = sorted(new_grad, reverse=True)[0:20]
    locations = sorted([np.where(new_grad == x)[0][0] for x in largest])

    if largest[0] < 2.5 * 10 ** (-10) or any(x >= 50 for x in np.gradient(locations)):
        #print("result:", 0)
        result = np.int64(0)
    else:
        #print("result:", locations[0] + offset)
        result = locations[0] + offset

    # farther_i = np.amax([max_grad_i, min_grad_i])
    if debug:
        return locations, largest

    return result


def averages_are_close(signal, start_is, end_is, averages=None, std_sensitivity=0.015):
    """check if the values of segments are close to eachother. previously calculated
    averages may also be included. difference in values is determined by
    calculating the standard deviation of all values."""
    if averages is None:
        averages = []

    if len(start_is) == 0:
        return False

    if len(start_is) == 1 and len(averages) == 0:
        return True

    for i in range(len(start_is)):
        segment = signal[start_is[i]: end_is[i]]
        av = np.mean(segment)
        averages.append(av)

    av_of_avs = sum(averages) / len(averages)
    std = np.std(averages) / abs(av_of_avs)
    return std <= std_sensitivity


def average_of_gradient(signal, start_i, end_i, offset_percentage=0.05):
    """calculate the average value of the gradient of a signal between
    start_i and end_i. start_i may be offset if the start of a segment
    needs to be excluded (due to how find_flat_segments works the start of
    a given segment may be slightly before the signal truly flattens)"""
    length = end_i - start_i
    offset = int(offset_percentage * length)
    segment = signal[start_i + offset: end_i]
    grad = np.gradient(segment)
    return np.mean(grad)


def uniq_filter_neo(signal, filter_i):
    """find segments where a certain value repeats. this filter ignores parts
    where the signal deviates from the unique value momentarily."""
    uniqs, indices, counts = np.unique(signal[:], return_index=True, return_counts=True)
    max_repeat = np.amax(counts)
    if max_repeat <= 10:
        return [], []
    uniq_is = np.where(counts == max_repeat)

    max_vals = uniqs[uniq_is]
    where_repeat = np.where(signal == max_vals[0])
    where_repeat = list(where_repeat[0])
    where_repeat = [x for x in where_repeat if x > filter_i]

    if len(where_repeat) == 0:
        return [], []

    seg_start = np.amin(where_repeat)
    seg_end = np.amax(where_repeat)

    return [[seg_start, seg_end]], [2]


def reformat_segments(start_is, end_is):
    """reformat segment start and end indices into a single list"""
    lst = []
    for i in range(len(start_is)):
        lst.append([start_is[i], end_is[i]])

    return lst


def find_flat_segments(signal, rel_sensitive_length=0.07, relative_sensitivity=0.02):
    """finds segments in the signal where the value stays approximately the same for long periods.
    returns lengths of segments, as well as their start and end indices.
    rel_sensitive_length determines how long a segment needs to be marked
    and relative_sensitivity determines how close the values need to be."""
    lengths = []
    start_is = []
    end_is = []
    lock_val = None  # subsequent values are compared to this value

    #sensitive_length = len(signal) * rel_sensitive_length
    sensitive_length = 200
    length = 1
    for i in range(len(signal)):
        val = signal[i]

        if lock_val is None:
            is_close = False
        else:
            #print("rel", abs(abs(val - lock_val) / lock_val), "abs", abs(val - lock_val))
            #is_close = abs(abs(val - lock_val) / lock_val) < relative_sensitivity
            is_close = abs(val - lock_val) <= 2*10**(-10)

        if not is_close or (is_close and i == len(signal) - 1):
            #print("cut")
            if length > sensitive_length:
                start_is.append(start_i)
                end_is.append(i)
                lengths.append(length)
            start_i = i
            length = 1
            lock_val = val

        if is_close:
            length += 1

    return lengths, start_is, end_is


def cal_seg_score_flat(signal, start_i, end_i, printer,
                       uniq_w=1.5, grad_sensitivity=0.5 * 10 ** (-13),
                       grad_w=10 ** 12, len_w=1, max_len=2900):
    """calculate a goodness value (a value determining how likely a flat segment
    has been wrongly detected by find_flat_segments) for a segment in a
    signal. a goodness value is increased if a segment has no clear trend,
    has a small fraction of unique values and is long.
    goodness > 1 -> bad
    goodness < 1 -> good/suspicious
    goodness < 0 -> very good"""
    segment = signal[start_i: end_i]
    uniqs = np.unique(segment)

    uniquevals = len(uniqs)
    totvals = len(segment)
    frac_of_uniq = 1 - uniquevals / totvals

    uniq_conf = uniq_w * frac_of_uniq

    grad_average = abs(average_of_gradient(signal, start_i, end_i))
    printer.extended_write("grad_average: ", grad_average)

    if grad_average < grad_sensitivity:
        grad_conf = 0
    else:
        grad_conf = - grad_w * grad_average

    rel_len = (end_i - start_i) / max_len
    sig_len = len(signal)
    # print(len(signal))

    if sig_len >= max_len/2:
        len_w = 1.5 * len_w
        grad_conf *= 1.5

    len_conf = rel_len * len_w

    printer.extended_write("uniq_conf:", uniq_conf, "grad_conf:", grad_conf, "len_conf:", len_conf)

    tot_conf = uniq_conf + grad_conf + len_conf
    printer.extended_write("tot_conf:", tot_conf)
    return tot_conf


def flat_filter(signal, printer, grad_sens=0.5 * 10 ** (-13)):
    """find segments where the value stays the same value for a long period.
    also recalculates the tail of the signal and calculates
    a confidence value for the segment"""
    lengths, start_is, end_is = find_flat_segments(signal)

    if len(start_is) == 0:
        return [], []

    final_i = end_is[len(end_is) - 1]
    seg_is = reformat_segments(start_is, end_is)

    printer.extended_write("number of segments found ", len(seg_is))

    # recheck tail
    if final_i != len(signal) - 1:
        tail_ave = [np.mean(signal[final_i:])]
    else:
        tail_ave = []

    close = averages_are_close(signal, start_is, end_is, averages=tail_ave)

    # if the averages of all segments are close to each other, they are combined
    # into one segment
    if close:
        if not tail_ave:
            seg_is = [[start_is[0], end_is[len(end_is) - 1]]]
        else:
            seg_is = [[start_is[0], len(signal) - 1]]

    comb_segs = combine_segments(seg_is)

    printer.extended_write("number of segments outputted", len(comb_segs))
    # print(comb_segs)

    confidences = []
    for segment in comb_segs:
        confidences.append(cal_seg_score_flat(signal, segment[0], segment[1], printer, grad_sensitivity=grad_sens))

    return comb_segs, confidences


def cal_seg_score_spike(gradient, spikes, all_diffs, printer, max_sensitivities=None,
                        n_sensitivities=None,
                        grad_sensitivity=2 * 10 ** (-13),
                        sdens_sensitivity=0.1):
    """calculate a goodness value for a segment found by find_spikes. the confidence
    depends on the steepness of the spikes and their number,
    average gradient of the segment and the density of spikes.
    returns both the confidence and a segment that starts
    at the first spike and ends at the last.
    goodness > 1 -> bad
    goodness < 1 -> good"""
    if n_sensitivities is None:
        n_sensitivities = [20, 100]

    if max_sensitivities is None:
        max_sensitivities = [1.5, 1, .5]

    n = len(spikes)

    if n <= 1:
        return [], None

    score = .5

    first_spike = spikes[0]
    seg_start = first_spike[0]
    last_spike = spikes[len(spikes) - 1]
    seg_end = last_spike[len(last_spike) - 1]
    seg_len = seg_end - seg_start

    max_diffs = []
    for i in range(n):
        diffs = all_diffs[i]
        max_diffs.append(np.amax(diffs))

    av_max = np.mean(max_diffs)

    # TEST DIFFS----------------------------------------
    if av_max >= max_sensitivities[0]:
        score += 2
    elif av_max >= max_sensitivities[1]:
        score += 1
    elif av_max >= max_sensitivities[2]:
        score += .5
    # --------------------------------------------------

    if n == 1:
        return [seg_start, seg_end], score

    spike_density = n / seg_len

    grad_ave = abs(np.mean(gradient[seg_start:seg_end]))

    # TEST NUMBER OF SPIKES-----------------------------
    if n >= n_sensitivities[1]:
        score += 1
    elif n >= n_sensitivities[0]:
        score += .5
    # --------------------------------------------------

    # TEST GRADIENT-------------------------------------
    if grad_ave >= grad_sensitivity:
        score -= .25
    else:
        score += .5
    # --------------------------------------------------

    # TEST SPIKE DENSITY--------------------------------
    if spike_density >= sdens_sensitivity:
        score += 1

    score = score / 1.5

    printer.extended_write("num_spikes", n, "av_diff", av_max, "grad_ave", grad_ave,
          "spike_density", spike_density, "badness", score)

    return [seg_start, seg_end], score


def find_spikes(gradient, filter_i, grad_sensitivity, len_sensitivity=6, start_sens=150):
    """find spikes in the signal by checking where the absolute gradient of the signal
    abruptly goes above grad_sensitivity. returns the spikes and their
    difference between the gradient and the grad_sensitivity"""
    spikes = []
    all_diffs = []

    diffs = []
    spike = []
    # print("filter i", filter_i)
    for i in range(filter_i, len(gradient)):
        val = abs(gradient[i])

        if val > grad_sensitivity:
            spike.append(i)
            diffs.append((val - grad_sensitivity) / grad_sensitivity)
            continue

        if i - 1 in spike:
            if len(spike) < len_sensitivity and abs(filter_i - spike[0]) > start_sens:
                spikes.append(spike)
                all_diffs.append(diffs)

            spike = []
            diffs = []

    return spikes, all_diffs


def spike_filter_neo(signal, filter_i, printer, grad_sensitivity=10 ** (-10)):
    """finds segments with steep spikes in the signal and calculates their goodness"""
    gradient = np.gradient(signal)
    spikes, all_diffs = find_spikes(gradient, filter_i, grad_sensitivity)
    seg_is, confidence = cal_seg_score_spike(gradient, spikes, all_diffs, printer)

    if len(seg_is) == 0:
        return [], []

    final_i = len(signal) - 1

    if seg_is[1] != final_i:
        tail = signal[seg_is[1]:]
        tail_ave = np.mean(tail)
        close = averages_are_close(signal, [seg_is[0]], [seg_is[1]], averages=[tail_ave])

        if close:
            seg_is[1] = final_i

    return [seg_is], [confidence]


def get_fft(signal, filter_i=0):
    """calculate the fft (absolute value) of the signal"""
    from scipy.fft import fft

    if len(signal) == 0:
         return [0]

    ftrans = fft(signal[filter_i:])
    # ftrans_abs = [abs(x) for x in ftrans]
    ftrans_abs = abs(ftrans)
    # ftrans_abs = ftrans
    return ftrans_abs, ftrans


def get_ffft(signal, k=2):
    sig_len = len(signal)
    const = 2 * np.pi / sig_len

    fft_sum = 0

    for n in range(sig_len):
        sig_val = signal[n]
        # print(sig_val)
        fft_real = np.cos(const * k * n)
        fft_imag = - np.sin(const * k * n)
        fft = sig_val * complex(fft_real, fft_imag)
        # print(fft_real, fft_imag, sig_val, fft)
        fft_sum += fft


    #print(fft_sum, abs(fft_sum))
    return fft_sum, abs(fft_sum)


#import math


def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.
    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.
    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.
    Example of usage :

        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2 ** 2 + d1 ** 2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results


def get_ffft_alt(signal, freq_range, magic_factor=1.5e7):
    # N = len(signal)
    #const = 2 * np.pi / N
    #x = list(range(N))
    #exp_list = [const * k * n for n in n_range]
    # reals = np.cos(exp_list)
    # imags = - np.sin(exp_list)
    # nu_reals = np.dot(signal, reals)
    # nu_imags = np.dot(signal, imags)
    # real_sum = np.sum(nu_reals)
    # imag_sum = np.sum(nu_imags)
    #abs_vals = np.sqrt(nu_reals ** 2 + nu_imags ** 2)
    # fin_list = np.dot(signal, abs_vals)
    #fft_sum = np.sum(abs_vals)
    # abs_sum = np.sqrt(real_sum ** 2 + imag_sum ** 2)

    # bin_freq = k * sample_freq / N
    freqs, results = goertzel(signal, sample_freq, freq_range)
    res = results[0]
    nu_abs = np.sqrt(res[0]**2 + res[1]**2)
    # nu_abs = res[0]
    # print(freqs, results, nu_abs)
    #return results[0][2] * magic_factor
    return  nu_abs


def calc_fft_indices(signal, printer, indices=None, window=400, smooth_window=401, filter_offset=0, goertzel=False):
    """calculate the fft for a windowed part of the signal. the window is scanned across
    the entire signal so that the fft values are as a function of time.
    returns the specified indices of the fft as a function of time.
    before calculating the ffts the trend of the signal is removed by first
    calculating a rolling average with a very large smoothing window and then
    getting the difference between the smoothed signal and the original."""
    if indices is None:
        indices = [1, 2, 6]

    if len(signal) < window:
        printer.extended_write("stopping fft, window larger than signal")
        return None, None, None, None, None

    if goertzel:
        goertzel_freqs = []

        for i in indices:
            freq_start = i * sample_freq / window
            goertzel_freqs.append((freq_start, freq_start + 1))

    sig_len = len(signal)
    ftrans_points = sig_len - window
    i_arr = np.zeros((len(indices), ftrans_points))

    # calculate smoothed signal
    offset = int(smooth_window / 2)
    smooth_signal = hf.smooth(signal, window_len=smooth_window)
    # CHECK AGAIN IF THIS IS USED SOMEWHERE
    smooth_x = [x - offset + filter_offset for x in list(range(len(smooth_signal)))]

    # remove trailing values for the smooth signal
    new_smooth = []
    for i in range(sig_len):
        new_smooth.append(smooth_signal[i + offset])

    # remove trend from original signal
    filtered_signal = [a - b for a, b in zip(signal, new_smooth)]

    # calculate ffts
    for i in range(ftrans_points):
        end_i = i + window
        signal_windowed = filtered_signal[i: end_i]

        if not goertzel:
            ftrans, ftrans_comp = get_fft(signal_windowed)

            for j in range(len(indices)):
                index = indices[j]
                # print("orig", ftrans_comp[index])
                i_arr[j][i] = ftrans[index]
                # i_arr[j][i] = ftrans_comp[index]
        else:
            for j in range(len(indices)):
                # index = indices[j]
                freq_range = goertzel_freqs[j]
                # fft_comp, fft_abs = get_ffft(signal_windowed, k=index)
                # fft_abs = get_ffft_alt(signal_windowed, k=index)
                fft_abs = get_ffft_alt(signal_windowed, freq_range)
                # print("new", fft_comp)
                i_arr[j][i] = fft_abs

    nu_x = list(range(ftrans_points))
    # nu_x = [x + filter_offset for x in nu_x]

    return i_arr, nu_x, smooth_signal, smooth_x, filtered_signal

def calc_dft_constants(k, N):
    const = -2 * np.pi / N
    dft_factors = []
    for n in range(N):
        dft_factors.append(complex(np.cos(const * k * n), np.sin(const * k * n)))

    return dft_factors

def make_window(i, signal, window=400):
    return signal[i:i + window]

def calc_dft_point(segment, dft_consts):
    dft_list = np.dot(segment, dft_consts)
    dft_sum = np.sum(dft_list)
    return dft_sum

def calc_fft_index_fast(signal, printer, window=400, smooth_window=401, filter_offset=0):
    sig_len = len(signal)
    ftrans_points = sig_len - window

    # calculate smoothed signal
    offset = int(smooth_window / 2)
    smooth_signal = hf.smooth(signal, window_len=smooth_window)
    # CHECK AGAIN IF THIS IS USED SOMEWHERE
    smooth_x = [x - offset + filter_offset for x in list(range(len(smooth_signal)))]

    # remove trailing values for the smooth signal
    new_smooth = []
    for i in range(sig_len):
        new_smooth.append(smooth_signal[i + offset])

    # remove trend from original signal
    filtered_signal = [a - b for a, b in zip(signal, new_smooth)]

    dft_consts = calc_dft_constants(2, window)

    window_func = np.vectorize(make_window, signature="(),(m)->(k)")
    windows = window_func(list(range(ftrans_points)), filtered_signal)

    dft_func = np.vectorize(calc_dft_point, signature="(n),(n)->()")
    fft_tseries = dft_func(windows, dft_consts)

    fft_tseries = abs(fft_tseries)

    return fft_tseries, list(range(ftrans_points)), smooth_signal, smooth_x, filtered_signal


def stats_from_i(i_arr, i_x, bad_segs, fft_window, printer, cut_length=70, max_sig_len=800, lol=False):
    """ status:
    0 = good
    1 = bad
    2 = undetermined"""
    printer.extended_write("filtering fft:")
    offset = int(len(i_arr) * 0.0875)
    filter_i_i = filter_start(i_arr, offset=offset, max_rel=.175)
    #filter_i_i = filter_i_list[0]

    last_i = len(i_arr) - 1

    if len(bad_segs) == 0:
        minus_i = cut_length
    else:
        final_i = bad_segs[0][0] - 2 * cut_length
        minus_i = last_i - final_i + fft_window

    nu_i_arr = i_arr[:-minus_i]
    cut_i_arr = nu_i_arr[filter_i_i:]
    nu_i_x = i_x[:-minus_i]

    i_arr_ave = np.mean(cut_i_arr)
    i_arr_sdev = np.std(cut_i_arr)

    grad = np.gradient(i_arr)
    cut_grad = grad[filter_i_i:-minus_i]
    grad_ave = np.mean(cut_grad)
    grad_x = list(range(filter_i_i, i_x[-1] + 1 - minus_i))

    grad_rmsd = np.sqrt(np.mean([(grad_ave - x) ** 2 for x in cut_grad]))

    printer.extended_write("i ave:", i_arr_ave, " i sdev:", i_arr_sdev)  # ave < 10e-09 - 5e-09 => SUS, sdev > 3e-09 => SUS
    printer.extended_write("grad rmsd", grad_rmsd)  # sus thresh: 2.5e-11, bad tresh: 7e-11 (raise this possibly)
    printer.extended_write("grad ave", grad_ave)  # > 1e-12 => SUS
    printer.extended_write("filtered fft length ", len(cut_i_arr))

    def change_status(new_stat, old_stat):
        if old_stat > 0:
            return old_stat

        return new_stat

    short = False
    status = 0
    sus_score = 0

    if len(cut_i_arr) < 400:
        printer.extended_write("SIGNAL TOO SHORT")
        status = change_status(3, status)
        short = True
        #return i_arr, nu_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short
        return nu_i_arr, nu_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short

    if len(cut_i_arr) < max_sig_len:
        printer.extended_write("NOT ENOUGH SIGNAL FOR ERROR LOCALIZATION")
        # status = change_status(3, status)
        short = True

    grad_ave_thresh = 2 * 10 ** (-12)
    # maybe 8.5e-12
    if grad_ave > 10 ** (-11):
        printer.extended_write("EXTREMELY HIGH GRADIENT AVERAGE")
        status = change_status(1, status)
    elif grad_ave > 3.5 * 10 ** (-12):
        printer.extended_write("INCREASING 50HZ")
        sus_score += 1

    if 3.5 * 10 ** (-11) < grad_rmsd < 2 * 10 ** (-10):
        printer.extended_write("SUSPICIOUS RMS")
        sus_score += 1
    elif grad_rmsd > 2 * 10 ** (-10):
        printer.extended_write("EXTREMELY HIGH RMS")
        status = change_status(1, status)

    if 3.5 * 10 ** (-9) < i_arr_sdev < 10 ** (-8):
        printer.extended_write("SUSPICIOUS SDEV")
        sus_score += 1
    elif i_arr_sdev > 10 ** (-8): # TODO increase maybe??
        printer.extended_write("EXTREMELY HIGH SDEV")
        status = change_status(1, status)

    # sus: 1e-08 - 5e-09, bad < 5e-09
    if i_arr_ave < 1 * 10 ** (-14): # 1.5e-09
        printer.extended_write("NO 50HZ DETECTED")
        status = change_status(1, status)
    elif i_arr_ave < 6 * 10 ** (-9):
        printer.extended_write("NOT ENOUGH 50HZ")
        status = change_status(2, status)
    elif i_arr_ave < 1.5 * 10 ** (-8):
        printer.extended_write("LOW 50HZ")
        sus_score += 1

    # TODO maybe dont do this?
    if sus_score >= 3:
        printer.extended_write("BAD SIGNAL")
        status = change_status(1, status)
    elif sus_score >= 2:
        printer.extended_write("SUSPICIOUS SIGNAL")
        status = change_status(2, status)

    #return i_arr, nu_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short
    return nu_i_arr, nu_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short


# TODO check that this works properly
def find_saturation_point_from_fft(i_x, i_arr, filter_i, fft_window, printer, sdev_window=10, rel_sdev_thresh=1.75,
                                   abs_sdev_thresh=1.35 * 10 ** (-10)):
    if len(i_arr) == 0:
        return None, None, None, None, None

    x_start_i = np.where(i_x == filter_i)[0][0]
    fft_sdev, rms_x = hf.averaged_signal(i_arr[filter_i:], sdev_window, i_x[x_start_i:], mode=2)

    sdev_mean = np.mean(fft_sdev)
    sdev_sdev = np.std(fft_sdev)
    sdev_thresh = sdev_mean + rel_sdev_thresh * sdev_sdev
    where_above_sdev = np.where(fft_sdev > sdev_thresh)[0]
    if len(where_above_sdev) != 0:
        sdev_span = [rms_x[where_above_sdev[0]], rms_x[where_above_sdev[-1]]]
        span_sdev_ave = np.mean(fft_sdev[where_above_sdev[0]:where_above_sdev[-1]])
        seg_len = hf.length_of_segments([sdev_span])
        highsdev = span_sdev_ave > abs_sdev_thresh
        # ave_diff = span_sdev_ave - sdev_mean
        #highsdev = ave_diff > 2.5*10**(-11)
        span_start_i = where_above_sdev[0]
        printer.extended_write("span_sdev_ave", span_sdev_ave, "seg_len", seg_len, "span_start_i", span_start_i)
        # printer.extended_write("ave diff", ave_diff)
        local_err = 200 <= seg_len <= 500 # TODO increase upper bound?

        # ave diff 2.5-6e-11
        if highsdev and local_err and span_start_i > 6:
            printer.extended_write("saturation point found")
            error_start = sdev_span[0] + fft_window + sdev_window
        else:
            printer.extended_write("no saturation point found")
            error_start = None

        return rms_x, fft_sdev, error_start, sdev_thresh, sdev_span

    return rms_x, fft_sdev, None, sdev_thresh, None


def fft_filter(signal, filter_i, bad_segs, printer, fft_window=400, indices=[2], badness_sens=.5, debug=False, fft_cut=70, min_length=400,
               goertzel=False, lol=False):

    normal_sig = signal[filter_i:]
    sig_len = len(normal_sig)
    final_i_filtsignal = sig_len - 1
    final_i_fullsignal = len(signal) - 1

    if len(bad_segs) == 0:
        bad_len = 0
        good_len = sig_len - fft_window - fft_cut
    else:
        bad_start_i = bad_segs[0][0] - 2 * fft_cut
        bad_len = final_i_filtsignal - bad_start_i
        minus_i = final_i_filtsignal - bad_start_i + fft_window
        good_len = len(normal_sig[:-minus_i])


    rel_bad_len = bad_len / sig_len

    if rel_bad_len >= badness_sens or good_len < min_length:
        printer.extended_write("NOT ENOUGH SIGNAL FOR FFT")
        if debug:
            return None, None, None, None, None, None, None, None, 2, 0, None, None, None, None, None, None
        else:
            return [], []

    #i_arr, i_x, smooth_signal, smooth_x, detrended_sig = calc_fft_indices(normal_sig, printer, indices=indices, window=fft_window,
                                                                          #filter_offset=filter_i, goertzel=goertzel)

    i_arr, i_x, smooth_signal, smooth_x, detrended_sig = calc_fft_index_fast(normal_sig, printer, filter_offset=filter_i)

    if i_arr is None:
        if debug:
            return None, None, None, None, None, None, None, None, 2, 0, None, None, None, None, None, None
        else:
            return [], []

    # cut_i_arr, cut_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short = stats_from_i(
    #     i_arr[0], i_x, bad_segs, fft_window, printer, cut_length=fft_cut, lol=lol)

    cut_i_arr, cut_i_x, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, short = stats_from_i(
         i_arr, i_x, bad_segs, fft_window, printer, cut_length=fft_cut, lol=lol)

    if not short:
        rms_x, fft_sdev, error_start, sdev_thresh, sdev_span = find_saturation_point_from_fft(cut_i_x, cut_i_arr,
                                                                                              filter_i_i,
                                                                                              fft_window, printer)
    else:
        rms_x, fft_sdev, error_start, sdev_thresh, sdev_span = None, None, None, None, None

    if debug:
        return cut_i_x, cut_i_arr, filter_i_i, i_arr_ave, i_arr_sdev, cut_grad, grad_ave, grad_x, status, sus_score, rms_x, fft_sdev, error_start, sdev_thresh, sdev_span, detrended_sig

    return score_fft_segment(status, error_start, final_i_fullsignal, filter_i_i)


def score_fft_segment(status, error_start, final_i_fullsignal, filter_i_i):
    if status == 0:
        score = -.5

    if status == 3:
        score = 0.0

    if status == 1:
        score = 1.5

    if status == 2:
        score = .5

    if error_start is not None:
        segments = [[error_start, final_i_fullsignal]]
        score = 1.5
    elif score > 0:
        segments = [[filter_i_i, final_i_fullsignal]]
    else:
        segments = []

    return segments, [score]


def analyze_fft(fft_i, window=400):
    start_i = 0
    end_i = window - 1

    final_i = len(fft_i) - 1

    sdev = []
    cont = True
    while cont:
        seg = fft_i[start_i:end_i]
        sdev.append(np.std(seg))

        start_i = end_i
        end_i = end_i + window

        if end_i > final_i:
            end_i = final_i

        if start_i >= final_i:
            cont = False

    return sdev


def find_default_y(arr, num_points=5000, step=.1 * 10 ** (-7)):
    """UNUSED
    this function would be used to find the default value of the fft function
    calculated by the calc_fft_indices function
    scan a signal across the y axis and find the highest value segment which
    contains a significant amount of all the values in a signal."""
    y_arr = np.linspace(0, 1 * 10 ** (-7), num_points)
    arr_len = len(arr)
    frac_arr = []

    # calculate the fraction of signals in a segment of length step as a
    # function of y
    for y_min in y_arr:
        y_max = y_min + step
        vals_in_step = [val for val in arr if y_min < val < y_max]
        frac = len(vals_in_step) / arr_len
        frac_arr.append(frac)

    # smooth the fraction function (this is done so that not every small
    # local maximum is detected)
    frac_arr = np.asarray(frac_arr)
    smooth_window = 201
    offset = int(smooth_window / 2)
    smooth_frac = hf.smooth(frac_arr, window_len=smooth_window)
    smooth_x = [x - offset for x in list(range(len(smooth_frac)))]

    new_smooth = []
    for i in range(num_points):
        new_smooth.append(smooth_frac[i + offset])

    # find local extrema (whose value is above .1) of the smoothed fraction function
    from scipy.signal import argrelextrema
    new_smooth = np.asarray(new_smooth)
    smooth_max_is = argrelextrema(new_smooth, np.greater, order=10)[0]

    maxima = new_smooth[smooth_max_is]
    maxima = [val for val in maxima if val > .1]
    max_is = []

    for i in range(len(new_smooth)):
        val = new_smooth[i]
        if val in maxima:
            max_is.append(i)

    # return the highest of these maxima
    if len(max_is) == 0:
        final_i = None
        max_step = np.amax(frac_arr)
        max_i = list(frac_arr).index(max_step)
        seg_min = y_arr[max_i]
        seg_max = seg_min + step
    else:
        final_i = np.amax(max_is)
        seg_min = y_arr[final_i]
        seg_max = seg_min + step

    return y_arr, frac_arr, (seg_min, seg_max), new_smooth, max_is, final_i


# TODO test this again
def get_spans_from_fft(fft_i2, hseg, printer, fft_window=400):
    """UNUSED
    this function would be used in conjunction with find_default_y to
    find the segments where the signal is no longer periodic."""
    segs = []
    all_fft_segs = []

    fft_segs = []
    temp_seg = [-1, -1]
    i_min = -1
    i_max = -1
    for fft_i in range(len(fft_i2)):
        fft_val = fft_i2[fft_i]
        val_in_hseg = hseg[0] < fft_val < hseg[1]

        if (i_max != -1 and i_min != - 1) and (not val_in_hseg or fft_i == len(fft_i2) - 1):
            printer.extended_write("in", i_min, i_max)
            if temp_seg[0] < i_min < temp_seg[1]:
                temp_seg = [temp_seg[0], i_max]
                printer.extended_write("between", temp_seg)
            elif temp_seg == [-1, -1]:
                temp_seg = [i_min, i_max]
                printer.extended_write("none", temp_seg)
            else:
                segs.append(temp_seg)
                printer.extended_write("append", temp_seg)
                temp_seg = [i_min, i_max]

            i_min = -1
            i_max = -1
            continue

        if val_in_hseg:
            i_min_temp = fft_i
            i_max_temp = fft_i + fft_window

            if i_min == -1:
                i_min = i_min_temp

            if i_max_temp > i_max:
                i_max = i_max_temp

    if temp_seg != [-1, -1]:
        printer.extended_write("final append")
        segs.append(temp_seg)

    return segs


def combine_segments(segments):
    """combine several segments so that there is no overlap between them"""
    n = len(segments)

    if n == 0:
        return []

    if n == 1:
        return segments

    segments_sorted = sorted(segments, key=itemgetter(0))

    combined_segs = []
    anchor_seg = segments_sorted[0]

    for i in range(1, n):
        segment = segments_sorted[i]

        if anchor_seg[1] < segment[0]:
            combined_segs.append(anchor_seg)
            anchor_seg = segment

        new_start = anchor_seg[0]
        new_end = max(anchor_seg[1], segment[1])

        anchor_seg = [new_start, new_end]

        if i == n - 1:
            combined_segs.append(anchor_seg)

    return combined_segs


def separate_segments(segments, confidences, conf_threshold=1):
    """sort segments into bad and suspicious based on their confidence values"""
    n = len(segments)

    bad_segs = []
    suspicious_segs = []

    for i in range(n):
        conf = confidences[i]
        segment = segments[i]

        if conf >= conf_threshold:
            bad_segs.append(segment)
        elif conf >= 0:
            suspicious_segs.append(segment)

    return bad_segs, suspicious_segs


def fix_overlap(bad_segs, suspicious_segs):
    """fix overlap between suspicious and bad segments. bad segments take priority
    over suspicious ones"""
    if len(bad_segs) == 0 or len(suspicious_segs) == 0:
        return suspicious_segs

    new_suspicious_segs = []
    for sus_seg in suspicious_segs:
        sus_list = list(range(sus_seg[0], sus_seg[1] + 1))
        for bad_seg in bad_segs:
            bad_list = list(range(bad_seg[0], bad_seg[1] + 1))
            sus_list = list(set(sus_list) - set(bad_list))

        split_lists = hf.split_into_lists(sus_list)
        split_segs = []

        for lst in split_lists:
            split_segs.append([np.amin(lst), np.amax(lst)])

        new_suspicious_segs += split_segs

    return new_suspicious_segs


def final_analysis(segments, confidences):
    """take all segments and their confidences and separate segments into good,
    suspicious and bad, as well as fix the overlap between them."""
    bad_segs, suspicious_segs = separate_segments(segments, confidences)

    bad_segs = combine_segments(bad_segs)
    suspicious_segs = combine_segments(suspicious_segs)

    suspicious_segs = fix_overlap(bad_segs, suspicious_segs)

    return bad_segs, suspicious_segs


def analyse_all_neo(signals, names, chan_num, printer,
                    filters=None,
                    filter_beginning=True,
                    fft_goertzel=False):
    """go through all signals and determine suspicious and bad segments within them.
    this is done by running the signal through three different filters
    (spike_filter_neo, flat_filter_neo, uniq_filter_neo and fft_filter).
    the function returns lists containing all bad and suspicious segments
    as well as ones containing whether the signal is bad (boolean value) and
    the time it took to analyse each signal."""
    if filters is None:
        filters = ["uniq", "flat", "spike", "fft"]

    # move fft filter to last place (requires other bad segments as input)
    if "fft" in filters:
        filters.append(filters.pop(filters.index("fft")))

    exec_times = []
    signal_statuses = []
    bad_segment_list = []
    suspicious_segment_list = []

    for i in range(chan_num):
        printer.extended_write(names[i])
        signal = signals[i]
        signal_length = len(signal)
        segments = []
        confidences = []
        # bad = False

        start_time = time.time()

        if filter_beginning:
            filter_i = filter_start(signal)
        else:
            filter_i = 0

        for fltr in filters:
            printer.extended_write("beginning analysis with " + fltr + " filter")

            if fltr == "uniq":
                seg_is, confs = uniq_filter_neo(signal, filter_i)

            if fltr == "flat":
                seg_is, confs = flat_filter(signal, printer)

            if fltr == "spike":
                seg_is, confs = spike_filter_neo(signal, filter_i, printer)

            if fltr == "fft":
                temp_bad_segs, temp_suspicious_segs = final_analysis(segments, confidences)
                seg_is, confs = fft_filter(signal, filter_i, temp_bad_segs, printer, goertzel=fft_goertzel)

            new_segs = len(seg_is)

            if new_segs == 0:
                printer.extended_write("no segments found")
            else:
                printer.extended_write(new_segs, "segment(s) found:")

                for seg in seg_is:
                    printer.extended_write(seg)

            segments += seg_is
            confidences += confs

        bad_segs, suspicious_segs = final_analysis(segments, confidences)
        num_bad = len(bad_segs)
        bad = num_bad > 0
        num_sus = len(suspicious_segs)
        printer.extended_write(num_sus, "suspicious and", num_bad, " bad segment(s) found in total")

        if not bad:
            printer.extended_write("no bad segments found")

        signal_statuses.append(bad)
        bad_segment_list.append(bad_segs)
        suspicious_segment_list.append(suspicious_segs)

        end_time = time.time()
        exec_time = end_time - start_time
        printer.extended_write("execution time:", exec_time)
        exec_times.append(exec_time)

        printer.extended_write()

    return signal_statuses, bad_segment_list, suspicious_segment_list, exec_times
