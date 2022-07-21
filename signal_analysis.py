import math

import ruptures as rpt
import numpy as np
import pandas as pd
import time
from operator import itemgetter

bkps = 8


def smooth(x, window_len=21, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

# fixes signals in bad formats (when reading only one channel)
def reorganize_signal(signal):
    size = len(signal)
    new_sig = np.zeros(size)
    # print(size)

    for i in range(size):
        new_sig[i] = signal[i][0]

    return new_sig

def find_nearby_detectors(d_name, detectors, r_sens = 0.06):
    dut = detectors[d_name]
    r_dut = dut[:3, 3]
    v_dut = dut[:3, 2]

    nears = []

    for name in detectors:
        if name == d_name:
            continue

        detector = detectors[name]
        r = detector[:3, 3]
        delta_r = np.sqrt((r_dut[0] - r[0])**2 +
                          (r_dut[1] - r[1])**2 +
                          (r_dut[2] - r[2])**2)

        if delta_r < r_sens:
            nears.append(name)

    return nears


# filter the jump in the beginning of the signal. works better on good signals
def filter_start(signal, offset=50, max_rel=0.05):
    max_i = int(max_rel * len(signal))
    grad = np.gradient(signal[:max_i])
    max_grad_i = np.argmax(grad)
    min_grad_i = np.argmin(grad)
    farther_i = np.amax([max_grad_i, min_grad_i])
    return farther_i + offset


# check if the values of segments are close to eachother.
def averages_are_close(signal, start_is, end_is, averages=[], std_sensitivity=0.015):
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


# UNUSED
def find_start_of_seg(signal, end_i):
    averages = np.zeros_like(signal)
    averages[end_i] = signal[end_i]

    for i in range(end_i - 1, -1, -1):
        seg = signal[i: end_i]
        ave = sum(seg) / len(seg)
        averages[i] = ave

    return averages


def average_of_gradient(signal, start_i, end_i, offset_percentage=0.05):
    length = end_i - start_i
    offset = int(offset_percentage * length)
    segment = signal[start_i + offset: end_i]
    grad = np.gradient(segment)
    return sum(grad) / len(grad)

def cal_confidence_seg(signal, start_i, end_i,
                       uniq_w=1.5, grad_sensitivity=0.5 * 10 ** (-13),
                       grad_w=10**12, len_w=1):
    segment = signal[start_i : end_i]
    uniqs = np.unique(segment)

    uniquevals = len(uniqs)
    totvals = len(segment)
    frac_of_uniq = 1 - uniquevals / totvals

    uniq_conf = uniq_w * frac_of_uniq

    grad_average = average_of_gradient(signal, start_i, end_i)

    if grad_average < grad_sensitivity:
        grad_conf = 0
    else:
        grad_conf = - grad_w * grad_average

    rel_len = (end_i - start_i) / len(signal)

    if rel_len >= .5:
        len_w = 1.5 * len_w

    len_conf = rel_len * len_w

    print("uniq_conf:", uniq_conf, "grad_conf:", grad_conf, "len_conf:", len_conf)

    tot_conf = uniq_conf + grad_conf + len_conf
    return tot_conf

#220
# finds segments in the signal where the value stays approximately the same for long periods
def find_uniq_segments(signal, rel_sensitive_length = 0.07, relative_sensitivity=0.02):
    lengths = []
    start_is = []
    end_is = []
    lock_val = None

    sensitive_length = len(signal) * rel_sensitive_length
    length = 1
    for i in range(len(signal)):
        val = signal[i]

        if lock_val == None:
            is_close = False
        else:
            is_close = abs(abs(val - lock_val) / lock_val) < relative_sensitivity

        if not is_close or (is_close and i == len(signal) - 1):
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

def uniq_filter_neo(signal, filter_i):
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

def reformat_stats(start_is, end_is):
    list = []
    for i in range(len(start_is)):
        list.append([start_is[i], end_is[i]])

    return list

def segment_filter_neo(signal):
    lengths, start_is, end_is = find_uniq_segments(signal)

    if len(start_is) == 0:
        return [], []

    final_i = end_is[len(end_is) - 1]
    seg_is = reformat_stats(start_is, end_is)

    #recheck tail
    if final_i != len(signal) - 1:
        tail_ave = [np.mean(signal[final_i:])]
    else:
        tail_ave = []

    close = averages_are_close(signal, start_is, end_is, averages=tail_ave)

    if close:
        seg_is = [[start_is[0], end_is[len(end_is) - 1]]]

    confidences = []
    #print(seg_is)
    for segment in seg_is:
        #print(segment)
        confidences.append(cal_confidence_seg(signal, segment[0], segment[1]))

    return seg_is, confidences

def cal_confidence_grad(gradient, spikes, all_diffs, max_sensitivities=[1.5, 1, .5],
                   n_sensitivities=[20, 100],
                   grad_sensitivity=2 * 10 ** (-13),
                        sdens_sensitivity=0.1):
    n = len(spikes)

    if n == 0:
        return [], None

    score = .5

    first_spike = spikes[0]
    seg_start = first_spike[0]
    last_spike = spikes[len(spikes) - 1]
    seg_end = last_spike[len(last_spike) - 1]
    seg_len = seg_end - seg_start

    if n == 1:
        return [seg_start, seg_end], .1

    spike_density = n / seg_len

    max_diffs = []
    for i in range(n):
        diffs = all_diffs[i]
        max_diffs.append(np.amax(diffs))

    av_max = np.mean(max_diffs)

    grad_ave = abs(np.mean(gradient[seg_start:seg_end]))

    # TEST DIFFS----------------------------------------
    if av_max >= max_sensitivities[0]:
        score += 2
    elif av_max >= max_sensitivities[1]:
        score += 1
    elif av_max >= max_sensitivities[2]:
        score += .5
    # --------------------------------------------------

    #TEST NUMBER OF SPIKES-----------------------------
    if n >= n_sensitivities[1]:
        score += 1
    elif n >= n_sensitivities[0]:
        score += .5
    #--------------------------------------------------

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

    print("num_spikes", n, "av_diff", av_max, "grad_ave", grad_ave,
          "spike_density", spike_density, "badness", score)

    return [seg_start, seg_end], score

def find_spikes(gradient, filter_i, grad_sensitivity, len_sensitivity=6):
    spikes = []
    all_diffs = []

    diffs = []
    spike = []
    for i in range(filter_i, len(gradient)):
        val = abs(gradient[i])

        if val > grad_sensitivity:
            spike.append(i)
            diffs.append((val - grad_sensitivity) / grad_sensitivity)
            continue

        if i - 1 in spike:
            if len(spike) < len_sensitivity:
                spikes.append(spike)
                all_diffs.append(diffs )

            spike = []
            diffs = []

    return spikes, all_diffs

def gradient_filter_neo(signal, filter_i, grad_sensitivity=10 ** (-10)):
    gradient = np.gradient(signal)
    spikes, all_diffs = find_spikes(gradient, filter_i, grad_sensitivity)
    seg_is, confidence = cal_confidence_grad(gradient, spikes, all_diffs)

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

#largest spike usually at 14, others at 42, 70, 127. (2795, 2767, 2739)
def fifty_hz_filter(signal, filter_i=0):
    from scipy.fft import fft

    if len(signal) == 0:
        return [0]

    ftrans = fft(signal[filter_i:])
    ftrans_abs = [x.real for x in ftrans]
    ftrans_abs[0] = 0
    return ftrans_abs

def fifty_hz_filter2(signal, win=10):
    ave = pd.Series(signal).rolling(win).mean()
    noise = []

    length = len(signal)
    offset = (win - 1)

    new_ave = []

    for val in ave:
        if math.isnan(val):
            continue
        new_ave.append(val)

    for i in range(length - offset):

        noise.append(signal[i] - new_ave[i])

    return noise, new_ave

def hz_func(x, a, b, c, d, e):
    frec = 2 * np.pi * (5 * 10 ** (-3))
    return a * np.sin(frec * x + e) + b * np.sin((3 * frec) * x + e) + c * np.exp(-d * x)

def make_fit(signal):
    from scipy.optimize import curve_fit
    length = len(signal)
    xdat = list(range(0, length))
    try:
        popt, pcov = curve_fit(hz_func, xdat, signal, maxfev=1400)
    except RuntimeError:
        print("optimal parameters not found")
        popt = []
        pcov = []

    return popt, pcov

def empty_params(num_params, params, sdevs):
    temp_params = []
    temp_sdevs = []
    for i in range(num_params):
        temp_params.append(0)
        temp_sdevs.append(0)

    params.append(np.asarray(temp_params))
    sdevs.append(np.asarray(temp_sdevs))


def grad_fit_analysis(signal, window=100, num_params=5):
    length = len(signal)
    signal_end = length - 1
    grad = np.gradient(signal)

    params = []
    sdevs = []

    for i in range(length):
        print(round(i/length, 2))
        window_end = i + window
        if window_end > signal_end:
            end_i = signal_end
        else:
            end_i = window_end

        if end_i - i <= 4:
            empty_params(num_params, params, sdevs)
            continue

        windowed_grad = grad[i:end_i]
        popt, pcov = make_fit(windowed_grad)

        if len(popt) == 0:
            empty_params(num_params, params, sdevs)
        else:
            params.append(popt)
            sdevs.append(np.sqrt(np.diag(pcov)))


    return params, sdevs


def get_regular_spans(extrema, extrem_grad, change_sensitivities=[25, 40],
                      span=10, num_sensitivity=4, offset=0):
    length = len(extrema)
    grad_len = len(extrem_grad)
    print(length)
    print(len(extrem_grad))

    if length < 2 or grad_len < 2:
        return []

    reg_grads = []
    segments = []

    start_i = -1
    end_i = -1

    num_extrema = 0

    for i in range(length):
        grad_val = extrem_grad[i]

        print(start_i, end_i, num_extrema)

        if not (grad_val > change_sensitivities[0] and grad_val < change_sensitivities[1]) \
                or i == length - 1:
            if start_i != -1 and end_i != -1:
                if num_extrema >= num_sensitivity:
                    segments.append([start_i - offset, end_i - offset])

                start_i = -1
                end_i = -1
                num_extrema = 0
            continue

        current = extrema[i]
        reg_grads.append(current)

        if start_i == -1:
            start_i = current - span

        if i == 0:
            prev = -1
        else:
            prev = extrema[i - 1]

        if prev in reg_grads:
            end_i = -1

        if end_i == -1:
            num_extrema += 1
            end_i = current + span

    print(segments)
    return segments

def combine_segments(segments):
    n = len(segments)

    if n == 0:
        return []

    segments_sorted = sorted(segments, key=itemgetter(0))

    combined_segs = []
    anchor_seg = segments_sorted[0]

    if n == 1:
        return segments

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

def length_of_segments(segments):
    tot_len = 0
    for segment in segments:
        length = segment[1] - segment[0]
        tot_len += length

    return tot_len

def split_into_lists(original_list):
    n = len(original_list)

    if n == 0:
        return original_list

    new_lists = []
    list = [original_list[0]]
    for i in range(1, n):
        integer = original_list[i]
        prev_int = integer - 1

        if prev_int not in list:
            new_lists.append(list)
            list = [integer]
        elif i == n - 1:
            list.append(integer)
            new_lists.append(list)
        else:
            list.append(integer)

    return new_lists

def fix_overlap(bad_segs, suspicious_segs):
    if len(bad_segs) == 0 or len(suspicious_segs) == 0:
        return suspicious_segs

    new_suspicious_segs = []
    for sus_seg in suspicious_segs:
        sus_list = list(range(sus_seg[0], sus_seg[1] + 1))
        for bad_seg in bad_segs:
            bad_list = list(range(bad_seg[0], bad_seg[1] + 1))
            sus_list = list(set(sus_list) - set(bad_list))

        split_lists = split_into_lists(sus_list)
        split_segs = []

        for lst in split_lists:
            split_segs.append([np.amin(lst), np.amax(lst)])

        new_suspicious_segs += split_segs

    return new_suspicious_segs

def final_analysis(signal_length, segments, confidences, badness_sensitivity=.8):
    bad_segs, suspicious_segs = separate_segments(segments, confidences)

    bad_segs = combine_segments(bad_segs)
    suspicious_segs = combine_segments(suspicious_segs)

    suspicious_segs = fix_overlap(bad_segs, suspicious_segs)

    tot_bad_length = length_of_segments(bad_segs)
    rel_bad_length = tot_bad_length / signal_length
    badness = rel_bad_length >= badness_sensitivity

    return badness, bad_segs, suspicious_segs

def analyse_all_neo(signals, names, chan_num,
                    filters=["uniq", "segment", "gradient"]):
    exec_times = []
    signal_statuses = []
    bad_segment_list = []
    suspicious_segment_list = []

    for i in range(chan_num):
        print(names[i])
        signal = signals[i]
        signal_length = len(signal)
        segments = []
        confidences = []
        bad = False

        start_time = time.time()
        filter_i = filter_start(signal)

        for filter in filters:
            print("beginning analysis with " + filter + " filter")

            if filter == "uniq":
                seg_is, confs = uniq_filter_neo(signal, filter_i)

            if filter == "segment":
                seg_is, confs = segment_filter_neo(signal)

            if filter == "gradient":
                seg_is, confs = gradient_filter_neo(signal, filter_i)

            new_segs = len(seg_is)

            if new_segs == 0:
                print("no segments found")
            else:
                print(new_segs, "segment(s) found")

            segments += seg_is
            confidences += confs
            bad, bad_segs, suspicious_segs = final_analysis(signal_length, segments, confidences)

            if bad:
                print("bad singal, stopping")
                break

        num_bad = len(bad_segs)
        num_sus = len(suspicious_segs)
        print(num_sus, "suspicious and", num_bad, " bad segment(s) found in total")

        if not bad:
            print("signal not marked as bad")

        signal_statuses.append(bad)
        bad_segment_list.append(bad_segs)
        suspicious_segment_list.append(suspicious_segs)

        end_time = time.time()
        exec_time = end_time - start_time
        print("execution time:", exec_time)
        exec_times.append(exec_time)

        print()

    return signal_statuses, bad_segment_list, suspicious_segment_list, exec_times

def pd_get_changes(signal):
    print(signal)

def find_changes_rpt(signal, method):
    if method == "Pelt":
        algo = rpt.Pelt().fit(signal)
        points = algo.predict(pen=0.01)

    if method == "Dynp":
        algo = rpt.Dynp().fit(signal)
        points = algo.predict(n_bkps=bkps)

    if method == "Binseg":
        algo = rpt.Binseg().fit(signal)
        points = algo.predict(n_bkps=bkps)

    if method == "Window":
        algo = rpt.Window().fit(signal)
        points = algo.predict(n_bkps=bkps)

    return points


def rpt_plot(signal, points):
    rpt.display(signal, points)
