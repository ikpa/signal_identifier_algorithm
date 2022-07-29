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
    fname = "sample_data31.npz"
    #fname = "many_failed.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    from scipy.signal import argrelextrema

    for i in range(n_chan):
        name = names[i]
        signal = signals[i]
        print(name)

        filter_i = sa.filter_start(signal)
        #segment = sa.uniq_filter_neo(signal, filter_i)[0][0]
        #filtered_signal = signal[segment[0]:segment[1]]
        filtered_signal = signal[filter_i:]
        extrema, extrem_grad, grad, grad_x, smooth_grad, smooth_x, offset = sa.get_extrema(filtered_signal, filter_i)

        #print(segments)
        segments, ext_lens = sa.find_regular_spans2(signal, filter_i)

        print(segments, ext_lens)
        print()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(signal)
        ax2.plot(grad_x, grad)
        ax2.plot(smooth_x, smooth_grad)
        #ax2.plot(smooth_grad)

        for extrem in extrema:
            x = smooth_x[extrem]
            ax2.axvline(x=x, linestyle="--", color="red")

        plot_spans(ax2, segments)

        if len(extrema) == len(extrem_grad):
            extrem_x = [x + offset for x in extrema]
            ax3.plot(extrem_x, extrem_grad, ".-")
            ax3.axhspan(25, 40, alpha=.5)

        plt.show()

def test_hz4():
    channels = ["MEG0624", "MEG0724", "MEG0531", "MEG0541",
                                       "MEG0634", "MEG0121"]
    fname = "many_failed.npz"
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
            ftrans = sa.get_fft(signal_windowed)
            max_fft = np.amax(ftrans)
            #min_fft = abs(np.amin(ftrans))
            maximums.append(max_fft)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        #ax2.set_ylim(-0.01 * 10 ** (-7), 2 * 10 ** (-7))
        ax1.plot(signal)
        ax2.plot(maximums)
        plt.show()

def animate_fft():
    def animate(i):
        if i + window > max_i:
            end_i = max_i
        else:
            end_i = i + window

        signal_windowed = signal[i: end_i]
        ftrans = sa.get_fft(signal_windowed)
        i1.append(ftrans[1])
        i2.append(ftrans[2])
        i6.append(ftrans[6])

        if i % 20 == 0:
            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax2.set_ylim(-1 * 10 ** (-7), 1.5 * 10 ** (-6))
            ax1.plot(plot_time, signal_windowed)
            ax2.plot(ftrans[0:10], ".-")
            ax3.plot(i1, label="index 1")
            ax3.plot(i2, label="index 2")
            ax3.plot(i6, label="index 6")
            ax3.legend()

    from matplotlib.animation import FuncAnimation
    fname = "sample_data30.npz"
    channels = ["MEG0121", "MEG1114"]
    # channels = ["MEG1114"]
    # fname = "many_failed.npz"
    # channels = ["MEG0131"]
    signals, names, time, n_chan = fr.get_signals(fname, channels=channels)
    signal = signals[0]

    window = 400
    plot_time = time[:window]
    time_0 = plot_time[0]
    plot_time = [x - time_0 for x in plot_time]
    max_i = len(signals[0]) - 1

    i1 = []
    i2 = []
    i6 = []

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ani = FuncAnimation(fig, animate, frames=len(signal) - window, interval=.001, repeat=False)
    plt.show()

def test_hz():

    from scipy.optimize import curve_fit
    #fname = "sample_data30.npz"
    fname = "many_failed.npz"
    #channels = ["MEG0121", "MEG1114", "MEG0311", "MEG331", "MEG0334"]
    #channels = ["MEG1114"]
    #fname = "many_failed.npz"
    #channels = ["MEG0131"]
    signals, names, time, n_chan = fr.get_signals(fname)

    window = 400
    plot_time = time[:window]
    time_0 = plot_time[0]
    plot_time = [x - time_0 for x in plot_time]
    max_i = len(signals[0]) - 1

    fft_indices = [2]
    num_indices = len(fft_indices)

    for i in range(n_chan):
        name = names[i]
        print(name)
        signal = signals[i]
        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]
        sig_len = len(filtered_signal)

        i_arr, smooth_signal, smooth_x, new_signal = sa.calc_fft_indices(filtered_signal, fft_indices)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.get_shared_x_axes().join(ax1, ax2)
        #ax3.get_shared_x_axes().join(ax3, ax4)
        ax1.plot(filtered_signal, label="original signal")
        #ax1.plot(signal_x, fit(np.asarray(signal_x), *popt), label="fitted signal")
        ax1.plot(smooth_x, smooth_signal, label="smooth signal")
        ax1.plot(new_signal, label="new signal")
        ax1.legend()

        colors = plt.cm.rainbow(np.linspace(0, 1, num_indices))
        for j, c in zip(range(num_indices), colors):
            index = fft_indices[j]
            dat = i_arr[j]
            y_arr, frac_arr, hseg, smooth_frac, smooth_maxima, final_i = sa.find_default_y(dat)
            #frac_grad = np.diff(frac_arr)
            ax2.plot(dat, label=str(index), color=c)
            ax2.axhspan(hseg[0], hseg[1], alpha=.5)
            ax3.plot(frac_arr)
            ax3.plot(smooth_frac)

            for maxi in smooth_maxima:
                if maxi == final_i:
                    color = "red"
                else:
                    color = "blue"
                ax3.axvline(maxi, linestyle="--", color=color)

            #ax4.plot(y_arr[:len(y_arr) - 1], frac_grad, ".-", color=c)

        #ax2.set_ylim(-1 * 10**(-8), 2.2 * 10 ** (-7))
        ax2.legend()
        ax2.grid()
        plt.title(name)

        print()

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
    test_hz()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
