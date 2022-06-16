import ruptures as rpt
import numpy as np
import pandas as pd

bkps=8

def reorganize_signal(signal):
    size = len(signal)
    new_sig = np.zeros(size)
    #print(size)

    for i in range(size):
        new_sig[i] = signal[i][0]

    return new_sig

def find_uniq_segments(signal, sensitive_length):
    vals = []
    lengths = []
    start_is = []
    end_is = []
    prevval = None

    length = 1
    for i in range(len(signal)):
        val = signal[i]

        if val != prevval or (val == prevval and i == len(signal) - 1):
            if length > sensitive_length:
                start_is.append(start_i)
                end_is.append(i)
                vals.append(val)
                lengths.append(length)
            start_i = i
            length = 1

        if val == prevval:
            length += 1

        prevval = val

    return vals, lengths, start_is, end_is

#0 = bad
#1 = ambiguous
#2 = good
def uniq_filter(signal, sensitivity = [0.15, 0.7]):
    uniquevals = len(np.unique(signal))
    totvals = len(signal)
    frac_of_uniq = uniquevals / totvals

    if frac_of_uniq <= sensitivity[0]:
        category = 0
    elif sensitivity[0] <= frac_of_uniq <= sensitivity[1]:
        category = 1
    else:
        category = 2

    return frac_of_uniq, category

def analyse_all(data):
    #names = data.names
    signals = data.data
    #print(np.shape(signals)[1])
    #chan_num = np.shape(signals)[1]
    chan_num = data.n_channels

    signal_status = []
    fracs = []
    for i in range(chan_num):
        #print(names[i])
        signal = signals[:,i]

        frac_of_uniq, category = uniq_filter(signal)

        signal_status.append(category)
        fracs.append(frac_of_uniq)

    return signal_status, fracs



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