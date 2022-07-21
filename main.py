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

def test_hz5():
    channels = ["MEG0624", "MEG0724", "MEG0531", "MEG0541",
                "MEG0634", "MEG0121"]
    fname = "many_failed.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    from scipy.signal import argrelextrema

    for i in range(n_chan):
        name = names[i]
        signal = signals[i]
        print(name)

        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]
        grad = np.gradient(filtered_signal)
        window = 51
        offset = int(window/2)
        smooth_grad = sa.smooth(grad, window_len=window)
        #smooth_x = np.linspace(0, len(filtered_signal) - 1, len(smooth_grad))
        smooth_x = [x - offset for x in list(range(len(smooth_grad)))]

        order = 10
        maxima = argrelextrema(smooth_grad, np.greater, order=order)[0]
        minima = argrelextrema(smooth_grad, np.less, order=order)[0]
        extrema = list(maxima) + list(minima)
        extrema.sort()
        #extrema = [smooth_x[i] for i in extrema]

        if len(maxima) > 1:
            extrem_grad = np.gradient(extrema)
        else:
            extrem_grad = []

        print(len(grad))
        print(len(smooth_grad))

        segments = sa.get_regular_spans(extrema, extrem_grad, offset=offset)

        print()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(filtered_signal)
        ax2.plot(grad)
        ax2.plot(smooth_x, smooth_grad)
        #ax2.plot(smooth_grad)

        for extrem in extrema:
            x = smooth_x[extrem]
            ax2.axvline(x=x, linestyle="--", color="red")

        plot_spans(ax2, segments)

        if len(extrema) == len(extrem_grad):
            extrem_x = [x - offset for x in extrema]
            ax3.plot(extrem_x, extrem_grad, ".-")

        plt.show()

def test_hz4():
    channels = ["MEG0624", "MEG0724", "MEG0531", "MEG0541",
                                       "MEG0634", "MEG0121"]
    fname = "many_many_successful2.npz"
    signals, names, time, n_chan = fr.get_signals(fname, channels)

    def plot_params(ax, params):
        params = np.asarray(params)
        n_params = len(params[0])
        print(len(params))
        for i in range(n_params):
            points = params[:, i]
            print(np.shape(points))
            ax.plot(points, label=str(i + 1))
            plt.legend()


    for i in range(n_chan):
        name = names[i]
        signal = signals[i]

        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]

        print(name)
        params, errs = sa.grad_fit_analysis(filtered_signal, window=100)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(filtered_signal)
        ax2.plot(np.gradient(filtered_signal))
        plot_params(ax3, params)
        plt.show()


#TODO periodicity = 203
def test_hz3():
    channels = ["MEG0624", "MEG0724", "MEG0531", "MEG0541",
                "MEG0634", "MEG0121"]
    fname = "many_many_successful2.npz"
    signals, names, time, n_chan = fr.get_signals(fname, channels)

    from scipy.optimize import curve_fit

    frec = 2 * np.pi * (5 * 10**(-3))

    def func(x, a, b, c, d, e):
        return a * np.sin(frec * x + e) + b * np.sin((3 * frec) * x + e) + c * np.exp(-d * x)

    for i in range(n_chan):
        signal = signals[i]
        name = names[i]
        print(name)

        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]
        length = len(filtered_signal)
        grad = np.gradient(filtered_signal)
        xdat = list(range(0, length))
        x = np.linspace(0, length, length)
        popt, pcov = curve_fit(func, xdat, grad, maxfev=100000)
        print(popt)
        print(np.sqrt(np.diag(pcov)))
        print()

        fit = func(x, *popt)

        difference = []

        for i in range(length):
            fit_val = fit[i]
            actual_val = grad[i]
            difference.append(actual_val - fit_val)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(filtered_signal)
        ax2.plot(grad)
        ax2.plot(fit)
        ax3.plot(difference)
        plt.show()


def test_hz2():
    from matplotlib.animation import FuncAnimation
    fname = datadir + "many_successful.npz"
    data = fr.load_all(fname).subpool(["MEG*1", "MEG*4"]).clip((0.210, 0.50))
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = sa.reorganize_signals(unorganized_signals, n_chan)

    window = 150

    for j in range(n_chan):
        signal = signals[j]
        print(names[j])
        signal_len = len(signal)
        max_i = signal_len - 1

        maximums = []
        for i in range(signal_len):
            if i + window > max_i:
                end_i = max_i
            else:
                end_i = i + window

            signal_windowed = signal[i: end_i]
            ftrans = sa.fifty_hz_filter(signal_windowed)
            max_fft = np.amax(ftrans)
            #min_fft = abs(np.amin(ftrans))
            maximums.append(max_fft)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        #ax2.set_ylim(-0.01 * 10 ** (-7), 2 * 10 ** (-7))
        ax1.plot(signal)
        ax2.plot(maximums)
        plt.show()



def test_hz():
    from matplotlib.animation import FuncAnimation
    fname = datadir + "sample_data20.npz"
    data = fr.load_all(fname).subpool(["MEG0311"]).clip((0.210, 0.50))
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = sa.reorganize_signals(unorganized_signals, n_chan)
    signal = signals[0]

    window = 1000
    max_i = len(signal) - 1

    print(range(0, max_i, window))

    signal_windowed = signal[0:window]
    ftrans = sa.fifty_hz_filter(signal_windowed)
    maximums = []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax2.set_ylim(-2*10**(-7), 2*10**(-7))

    def animate(i):
        #i = i * 10
        if i + window > max_i:
            end_i = max_i
        else:
            end_i = i + window

        signal_windowed = signal[i: end_i]
        ftrans = sa.fifty_hz_filter(signal_windowed)
        max_fft = np.amax(ftrans)
        min_fft = abs(np.amin(ftrans))
        maximums.append(max(max_fft, min_fft))

        if i % 10 == 0:
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax2.set_ylim(-1 * 10 ** (-7), 1 * 10 ** (-7))
            ax3.set_ylim(-0.5 * 10 ** (-7), 1 * 10 ** (-7))
            ax1.plot(signal_windowed)
            ax2.plot(ftrans)
            ax3.plot(maximums)



    ani = FuncAnimation(fig, animate, frames=len(signal), interval=.01, repeat=True)
    plt.show()


def secondver():
    fname = "sample_data02.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

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
    test_hz5()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
