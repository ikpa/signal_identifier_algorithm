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
    #plot_in_order(signals, names, n_chan, np.full((n_chan,2), (False, 2)), [], [], [], ylims=[-.2 * 10**(-8), 3.2 * 10 ** (-8)])

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
    confidences = []

    for i in range(len(signals)):
        segments = []
        seg_confidences = []
        signal = signals[i]
        name = names[i]
        print(name)
        filter_i = sa.filter_start(signal)

        where_repeat, conf = sa.uniq_filter_neo(signal, filter_i)
        segments += where_repeat
        seg_confidences += conf

        where_repeat, conf = sa.segment_filter_neo(signal)
        segments += where_repeat
        seg_confidences += conf

        where_repeat, conf = sa.gradient_filter_neo(signal, filter_i)
        segments += where_repeat
        seg_confidences += conf

        print(segments)
        #print(seg_confidences)
        seg_i_list.append(sa.combine_segments(segments))
        #confidences.append(seg_confidences)
        print(segments)
        print()

    plot_in_order_neo(signals, names, n_chan, bads, seg_is=seg_i_list, confidence_list=confidences)

def plot_in_order_neo(signals, names, n_chan, statuses, confidence_list,
                      seg_is, exec_times=[], ylims=None):
    print(confidence_list)
    for i in range(n_chan):
        name = names[i]
        signal = signals[i]
        bad = statuses[i] if not len(statuses) == 0 else False
        segments = seg_is[i] if not len(seg_is) == 0 else []
        exec_time = exec_times[i] if not len(exec_times) == 0 else 0
        confidences = confidence_list[i] if not len(confidence_list) == 0 else []

        if bad:
            #print("bad, skipping")
            #print()
            #continue
            status = "bad"
        else:
            status = "good"

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(signal)
        ax1.axvline(x=sa.filter_start(signal), linestyle="--")

        if len(segments) != 0:
            for j in range(len(segments)):
                segment = segments[j]
                if len(segment) == 0:
                    continue
                seg_start = segment[0]
                seg_end = segment[1]
                ax1.axvspan(seg_start, seg_end, alpha=.5)

                if confidences != []:
                    confidence = confidences[j]

                    if confidence is not None:
                        seg_len = seg_end - seg_start
                        x_pos = seg_start + 0.1 * seg_len
                        y_pos = 0.9 * np.amax(signal)
                        ax1.text(x_pos, y_pos, s=str(round(confidence, 2)), fontsize="xx-small")

        if ylims != None:
            ax1.set_ylim(ylims)

        ax2.plot(np.gradient(signal))
        title = name + ": " + status
        plt.title(title)
        plt.show()

def plot_spans(ax, segments, color="blue"):
    if len(segments) == 0:
        return

    for segment in segments:
        ax.axvspan(segment[0], segment[1], color=color, alpha=.5)

    return

def plot_in_order_ver3(signals, names, n_chan, statuses,
                       bad_seg_list, suspicious_seg_list, exec_times,
                       ylims=None):
    for i in range(n_chan):
        name = names[i]
        signal = signals[i]
        bad = statuses[i]
        bad_segs = bad_seg_list[i]
        suspicious_segs = suspicious_seg_list[i]
        exec_time = exec_times[i] if len(exec_times) != 0 else 0

        if bad:
            status = "bad"
        else:
            status = "good"

        fig, ax = plt.subplots()
        ax.plot(signal)

        plot_spans(ax, bad_segs, color="red")
        plot_spans(ax, suspicious_segs, color="yellow")

        ax.grid()
        title = name + ": " + status
        ax.set_title(title)
        plt.show()

def test_hz():
    fname = datadir + "many_many_successful.npz"
    data = fr.load_all(fname).subpool(["MEG*1", "MEG*4"]).clip((0.210, 0.50))
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = sa.reorganize_signals(unorganized_signals, n_chan)

    for i in range(len(signals)):
        signal = signals[i]
        name = names[i]
        print(name)

        filter_i = sa.filter_start(signal)
        ftrans = sa.fifty_hz_filter(signal, filter_i)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(signal[filter_i:])
        ax2.plot(ftrans)
        plt.title(name)
        plt.show()

def secondver():
    fname = datadir + "sample_data02.npz"
    data = fr.load_all(fname).subpool(["MEG*1", "MEG*4"]).clip((0.210, 0.50))
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = sa.reorganize_signals(unorganized_signals, n_chan)

    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(signals, names, n_chan)
    plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, exec_times)

def overlap():
    sus_segs = [[600, 1400]]
    bad_segs = [[600, 650], [900, 950], [1000, 1300]]
    sa.fix_overlap(bad_segs, sus_segs)


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
    statuses, bad_segs, sus_segs, exec_times = sa.analyse_all_neo(signals, names, n)
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

        if bads[i]:
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

if __name__ == '__main__':
    #basic()
    #analysis()
    #dataload()
    #averagetest()
    #firstver()
    #secondver()
    #plottest()
    #animtest()
    #simo()
    #nearby()
    #names()
    #simulation()
    #test_uniq()
    #overlap()
    test_hz()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
