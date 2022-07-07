import signal_generator as sg
import signal_analysis as sa
import matplotlib.pyplot as plt
import file_reader as fr
import numpy as np
import pandas as pd
import helmet_vis as vis
from mayavi import mlab
import re

methods = ["Pelt", "Dynp", "Binseg", "Window"]
datadir = "example_data_for_patrik/"

def plot_in_order(signals, names, n_chan, statuses, fracs=[], seg_is=[],
                  exec_times=[], ylims=None):
    for i in range(n_chan):
        name = names[i]
        signal = signals[i]
        bad = statuses[i]
        frac = fracs[i] if not len(fracs) == 0 else 0
        uniq_stats = seg_is[i] if not len(seg_is) == 0 else []
        exec_time = exec_times[i] if not len(exec_times) == 0 else 0

        #print(name)
        if bad[0]:
            #print("bad, skipping")
            #print()
            #continue
            status = "bad"
        else:
            status = "good"

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(signal)
        ax1.axvline(x=sa.filter_start(signal), linestyle="--")

        if ylims != None:
            ax1.set_ylim(ylims)

        ax2.plot(np.gradient(signal))
        title = name + ": " + status
        if not len(uniq_stats) == 0:

            #print(uniq_stats[0])
            if not len(uniq_stats[0]) == 0:
                #same_sum = uniq_stats[2] - uniq_stats[1]
                #same_frac = same_sum / len(signal)
                for j in range(len(uniq_stats)):
                    ax1.axvspan(float(uniq_stats[j][0]), float(uniq_stats[j][1]), alpha=.5)

        title += ", badness: " + str(bad[1])

        plt.title(title)
        plt.grid()

        plt.show()

def firstver():
    fname = datadir + "many_failed.npz"
    data = fr.load_all(fname).subpool(["MEG*1", "MEG*4"]).clip((0.210, 0.50))
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = sa.reorganize_signals(unorganized_signals, n_chan)
    statuses, fracs, uniq_stats_list, exec_times = sa.analyse_all(signals, names, n_chan)

    plot_in_order(signals, names, n_chan, statuses, fracs, uniq_stats_list, exec_times)

def simulation():
    n = 290
    x = np.linspace(0, n, n)
    n_chan = 20

    signals = []
    names = []

    for i in range(n_chan):
        name = str(i + 1)
        print(name)
        names.append(name)
        signal = sg.simple_many_flat(x, n, 4, flats_same=True)
        signals.append(signal)
        frac, indices, counts, i_arr, bad = sa.uniq_filter(signal)
        print("frac", frac)
        print("indices", indices)
        print("counts", counts)
        print("i_arr", i_arr)
        print()



    #statuses, fracs, uniq_stats_list, exec_times = sa.analyse_all(signals, names, n_chan)
    plot_in_order(signals, names, n_chan, np.full((n_chan,2), (False, 2)), [], [], [], ylims=[-.2 * 10**(-8), 3.2 * 10 ** (-8)])

def test_uniq():
    fname = datadir + "many_failed.npz"
    data = fr.load_all(fname).subpool(["MEG*1", "MEG*4"]).clip((0.210, 0.50))
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = sa.reorganize_signals(unorganized_signals, n_chan)

    seg_i_list = []
    bads = []

    for i in range(len(signals)):
        signal = signals[i]
        name = names[i]
        print(name)
        filter_i = sa.filter_start(signal)
        seg_is, bad = sa.gradient_filter(signal, filter_i)

        seg_i_list.append(seg_is)
        bads.append(bad)



    plot_in_order(signals, names, n_chan, bads, seg_is=seg_i_list)

def basic():
    signal, t, i = sg.gen_signal(containsNoise=True, containsError=True)
    signal_smooth = pd.Series(signal).rolling(window=10).mean()
    dsig = np.gradient(np.gradient(signal))
    dsig_smooth = pd.Series(dsig).rolling(window=7).mean()
    # dsig_smooth = np.gradient(signal_smooth)

    error_t = t[i]
    # print(error_t)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t, signal, label="normal")
    ax1.plot(t, signal_smooth, label="filtered")

    ax2.plot(t, dsig, label="normal")
    ax2.plot(t, dsig_smooth, label="filtered")
    if i != None:
        ax1.axvline(x=error_t, linestyle="--", c="black")
        ax2.axvline(x=error_t, linestyle="--", c="black")
    plt.legend()
    plt.show()

def plottest():
    fname = datadir + "many_successful.npz"
    data = fr.load_all(fname).subpool(["MEG*1", "MEG*4"]).clip((0.210, 0.50))
    unorganized_signals = data.data
    names = data.names
    n = data.n_channels
    signals = sa.reorganize_signals(unorganized_signals, n)
    detecs = np.load("array120_trans_newnames.npz")

    names, signals = order_lists(detecs, names, signals)

    #n = data.n_channels
    #signals = sa.reorganize_signals(signals, n)
    statuses, fracs, uniq_stats_list, exec_times = sa.analyse_all(signals, names, n)
    bad_list = bad_list_for_anim(names, statuses)

    vis.helmet_animation(names, signals, frames=1000, bads=bad_list)

def regex_filter(npz_data):
    names = []
    signals = []
    bads = []

    for item in npz_data:
        if item.endswith("1"):
            names.append(item)
            signals.append(npz_data[item])
        else:
            bads.append(item)


    return signals, names, bads

def bad_list_for_anim(names, bads):
    bad_names = []
    for i in range(len(names)):

        if bads[i][0]:
            bad_names.append(names[i])

    return bad_names

def simo():
    detecs = np.load("array120_trans_newnames.npz")
    #print(detecs)
    #print(detecs)
    #chan_num = len(names)
    #statuses = np.full((len(names), 2), (False, 2))
    signals = sg.simulate_eddy(detecs)
    signal_len = len(signals[0])
    # statuses, fracs, uniq_stats_list, exec_times = sa.analyse_all(signals, names, chan_num)
    # plot_in_order(signals,names, chan_num, statuses,
    #               fracs=fracs, uniq_stats_list=uniq_stats_list,
    #               exec_times=exec_times)
    vis.helmet_animation(detecs, signals, 1000)

def nearby():
    detecs = np.load("array120_trans_newnames.npz")
    dut = "MEG1011"
    nears = sa.find_nearby_detectors(dut, detecs)
    print(nears)
    matrixes, names, bads = regex_filter(detecs)
    bads.append(nears)
    print(bads)

    vis.plot_all(names, np.full(np.shape(names), 1), nears)

def order_lists(pos_list, dat_names, signals):
    new_signals = []
    new_names = []

    n = 0
    for name in pos_list:
        i = dat_names.index(name)
        new_names.append(dat_names[i])
        new_signals.append(signals[i])
        n += 1

    return new_names, new_signals


def nameorder():
    fname = datadir + "many_many_successful2.npz"
    data = fr.load_all(fname).subpool(["MEG*1", "MEG*4"]).clip((0.210, 0.50))

    detecs = np.load("array120_trans_newnames.npz")

    names_dat, signals = order_lists(detecs, data)

    i = 0
    for detec in detecs:
        print("data: " + names_dat[i])
        print("pos: " + detec)
        print()
        i += 1

if __name__ == '__main__':
    #basic()
    #analysis()
    #dataload()
    #averagetest()
    firstver()
    #plottest()
    #animtest()
    #simo()
    #nearby()
    #names()
    #simulation()
    #test_uniq()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
