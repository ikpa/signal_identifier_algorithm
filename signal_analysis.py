import ruptures as rpt
import numpy as np
import pandas as pd
import time

bkps=8

#fixes signals in bad formats (when reading only one channel)
def reorganize_signal(signal):
    size = len(signal)
    new_sig = np.zeros(size)
    #print(size)

    for i in range(size):
        new_sig[i] = signal[i][0]

    return new_sig

def reorganize_signals(signals, n):
    new_signals = []
    for i in range(n):
        signal = signals[:, i]
        new_signals.append(signal)

    return new_signals

#filter the jump in the beginning of the signal. works better on good signals
def filter_start(signal, offset=20):
    grad = np.gradient(signal)
    max_grad_i = np.argmax(grad)
    min_grad_i = np.argmin(grad)
    farther_i = np.amax([max_grad_i, min_grad_i])
    return farther_i + offset

#check if the values of segments are close to eachother.
def averages_are_close(signal, start_is, end_is, averages = [], std_sensitivity = 10**(-10)):
    if len(start_is) == 1 and len(averages) == 0:
        return True

    print("start is", len(start_is))
    #print(start_is)
    for i in range(len(start_is)):
        #print(i)
        segment = signal[start_is[i] : end_is[i]]
        av = sum(segment) / len(segment)
        #print(averages)
        averages.append(av)

    print(averages)
    av_of_avs = sum(averages) / len(averages)
    std = np.std(averages)/abs(av_of_avs)
    print("std", std)

    std_sensitivity = 0.01
    return std <= std_sensitivity

#UNUSED
def find_start_of_seg(signal, end_i):
    averages = np.zeros_like(signal)
    averages[end_i] = signal[end_i]

    for i in range(end_i - 1, -1, -1):
        seg = signal[i : end_i]
        ave = sum(seg) / len(seg)
        averages[i] = ave

    return averages

def average_of_gradient(signal, start_i, end_i, offset_percentage = 0.05):
    length = end_i - start_i
    offset = int(offset_percentage * length)
    segment = signal[start_i + offset: end_i]
    grad = np.gradient(segment)
    return sum(grad) / len(grad)

def cal_confidence(grad_average, rel_len, rel_dist,
                   len_w=1.5, grad_w=10**12, dist_w=1,
                   grad_lock = 0.5*10**(-13)):
    if grad_average < grad_lock:
        grad_conf = 0
    else:
        grad_conf = - grad_w * grad_average

    len_conf = len_w * rel_len
    dist_conf = - dist_w * rel_dist

    print("grad conf", grad_conf)
    print("len conf", len_conf)
    print("dist conf", dist_conf)

    confidence = grad_conf + len_conf + dist_conf

    return confidence

#the proper one for now
#further analyse segments which stay around one value for too long
def multi_seg_analysis(signal, start_is, end_is, badness_sensitivity):
    final_i = len(signal) - 1
    final_segment_i = end_is[len(end_is) - 1]

    end_of_segment = final_segment_i

    if final_i != final_segment_i:
        segment = signal[final_segment_i:]
        ave = sum(segment) / len(segment)

        if averages_are_close(signal, start_is, end_is, averages=[ave]):
            end_of_segment = final_i

    tot_length = end_of_segment - start_is[0]
    same_frac = tot_length / len(signal)
    bad = same_frac > badness_sensitivity
    print("same_frac", same_frac)
    #print("bad", bad)

    if bad:
        return (bad, 2), end_of_segment

    grad_average = average_of_gradient(signal, start_is[0], end_of_segment)
    rel_len = (end_of_segment - start_is[0]) / len(signal)
    rel_dist = (final_i - end_of_segment) / len(signal)
    confidence = cal_confidence(grad_average, rel_len, rel_dist)

    return (bad, confidence), end_of_segment

#finds segments in the signal where the value stays approximately the same for long periods
def find_uniq_segments(signal, sensitive_length = 220, relative_sensitivity = 0.02):
    lengths = []
    start_is = []
    end_is = []
    lock_val = None

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

def segment_filter(signal, badness_sensitivity = 0.8):
    lengths, start_is, end_is = find_uniq_segments(signal)

    same_sum = sum(lengths)
    same_frac = same_sum / len(signal)

    prebad = same_frac > badness_sensitivity

    #TODO signal considered bad even if segments dont have similar averages
    #fix (?)
    if not prebad and len(start_is) != 0:
        print("more than one uniq segment, checking similarity")

        if averages_are_close(signal, start_is, end_is, averages=[]):
            print("averages close, doing multiseg")
            bad, new_end = multi_seg_analysis(signal, start_is, end_is, badness_sensitivity)
            start_is = [start_is[0]]
            end_is = [new_end]
        else:
            print("averages not close, no multiseg")
            bad = (prebad, 2)

    else:
        print("no multiseg")
        bad = (prebad, 2)


    #TODO broken, fix
    return [lengths, start_is, end_is], bad



#check the percentage of the signal where the value stays the same
def uniq_filter(signal, sensitivity = 0.2):
    uniquevals = len(np.unique(signal))
    totvals = len(signal)
    frac_of_uniq = uniquevals / totvals

    return frac_of_uniq, (frac_of_uniq <= sensitivity, 2)

def analyse_all(signals, names, chan_num):

    exec_times = []
    signal_status = []
    fracs = []
    uniq_stats_list = []
    for i in range(chan_num):
        print(names[i])
        signal = signals[i]

        start_time = time.time()
        frac_of_uniq, bad = uniq_filter(signal)

        if not bad[0]:
            segment_stats, bad = segment_filter(signal)
        else:
            segment_stats = []

        end_time = time.time()

        print()

        exec_times.append(end_time - start_time)
        signal_status.append(bad)
        fracs.append(frac_of_uniq)
        uniq_stats_list.append(segment_stats)

    print(signal_status)
    print(uniq_stats_list)

    return signal_status, fracs, uniq_stats_list, exec_times

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