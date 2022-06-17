import ruptures as rpt
import numpy as np
import pandas as pd
import time

bkps=8

def reorganize_signal(signal):
    size = len(signal)
    new_sig = np.zeros(size)
    #print(size)

    for i in range(size):
        new_sig[i] = signal[i][0]

    return new_sig

def find_uniq_segments(signal, sensitive_length = 220, relative_sensitivity = 0.015,
                       badness_sensitivity = 0.8):
    vals = []
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
                vals.append(val)
                lengths.append(length)
            start_i = i
            length = 1
            lock_val = val

        if is_close:
            length += 1

    same_sum = sum(lengths)
    same_frac = same_sum / len(signal)

    bad = same_frac > badness_sensitivity

    return [vals, lengths, start_is, end_is], bad

def uniq_filter(signal, sensitivity = 0.2):
    uniquevals = len(np.unique(signal))
    totvals = len(signal)
    frac_of_uniq = uniquevals / totvals

    return frac_of_uniq, frac_of_uniq <= sensitivity

def analyse_all(data):
    signals = data.data
    chan_num = data.n_channels

    exec_times = []
    signal_status = []
    fracs = []
    uniq_stats_list = []
    for i in range(chan_num):
        signal = signals[:,i]

        start_time = time.time()
        frac_of_uniq, bad = uniq_filter(signal)

        if not bad:
            uniq_stats, bad = find_uniq_segments(signal)
        else:
            uniq_stats = []

        end_time = time.time()

        exec_times.append(end_time - start_time)
        signal_status.append(bad)
        fracs.append(frac_of_uniq)
        uniq_stats_list.append(uniq_stats)

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