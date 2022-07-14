import ruptures as rpt
import numpy as np
import pandas as pd
import time
from operator import itemgetter

bkps = 8

# fixes signals in bad formats (when reading only one channel)
def reorganize_signal(signal):
    size = len(signal)
    new_sig = np.zeros(size)
    # print(size)

    for i in range(size):
        new_sig[i] = signal[i][0]

    return new_sig

def reorganize_signals(signals, n):
    new_signals = []
    for i in range(n):
        signal = signals[:, i]
        new_signals.append(signal)

    return new_signals

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

    print("checking averages")

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

    print("uniq_conf:", uniq_conf)
    print("grad_conf:", grad_conf)
    print("len_conf:", len_conf)

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

    print("num_spikes", n, "av_diff", av_max, "grad_ave", grad_ave,
          "spike_density", spike_density)

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

    print("score", score)

    return [seg_start, seg_end], score / 1.5

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

#TODO find way to include tail in segments
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

#TODO make it so theres no overlapping sus and bad segments
def final_analysis(signal_length, segments, confidences, badness_sensitivity=.8):
    bad_segs, suspicious_segs = separate_segments(segments, confidences)

    bad_segs = combine_segments(bad_segs)
    suspicious_segs = combine_segments(suspicious_segs)

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

            segments += seg_is
            confidences += confs
            bad, bad_segs, suspicious_segs = final_analysis(signal_length, segments, confidences)

            if bad:
                print("bad singal, stopping")
                break

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
