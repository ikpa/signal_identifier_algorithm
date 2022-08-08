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

    # statuses, fracs, uniq_stats_list, exec_times = sa.analyse_all(signals, names, n_chan)
    # plot_in_order(signals, names, n_chan, np.full((n_chan,2), (False, 2)), [], [], [], ylims=[-.2 * 10**(-8), 3.2 * 10 ** (-8)])


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
        # print(seg_confidences)
        seg_i_list.append(sa.combine_segments(segments))
        # confidences.append(seg_confidences)
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
            # print("bad, skipping")
            # print()
            # continue
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
    # fname = "many_failed.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    from scipy.signal import argrelextrema

    for i in range(n_chan):
        name = names[i]
        signal = signals[i]
        print(name)

        filter_i = sa.filter_start(signal)
        # segment = sa.uniq_filter_neo(signal, filter_i)[0][0]
        # filtered_signal = signal[segment[0]:segment[1]]
        filtered_signal = signal[filter_i:]
        extrema, extrem_grad, grad, grad_x, smooth_grad, smooth_x, offset = sa.get_extrema(filtered_signal, filter_i)

        # print(segments)
        segments, ext_lens = sa.find_regular_spans2(signal, filter_i)

        print(segments, ext_lens)
        print()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(signal)
        ax2.plot(grad_x, grad)
        ax2.plot(smooth_x, smooth_grad)
        # ax2.plot(smooth_grad)

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


# TODO periodicity = 203
def test_hz3():
    channels = ["MEG0624", "MEG0724", "MEG0531", "MEG0541",
                "MEG0634", "MEG0121"]
    fname = "many_many_successful2.npz"
    signals, names, time, n_chan = fr.get_signals(fname, channels)

    from scipy.optimize import curve_fit

    frec = 2 * np.pi * (5 * 10 ** (-3))

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
            # min_fft = abs(np.amin(ftrans))
            maximums.append(max_fft)

        fig, (ax1, ax2) = plt.subplots(2, 1)
        # ax2.set_ylim(-0.01 * 10 ** (-7), 2 * 10 ** (-7))
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


def calc_all_fft_comp(signals, names, detecs):
    x = []
    y = []
    z = []

    window = 400
    nchan = len(signals)

    all_u = []
    all_v = []
    all_w = []
    for i in range(nchan):
        name = names[i]
        print(name)
        signal = signals[i]
        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]
        loc_matrx = detecs[name]
        r = loc_matrx[:3, 3]
        vec = loc_matrx[:3, 2]

        x.append(r[0])
        y.append(r[1])
        z.append(r[2])

        fft_i2, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filtered_signal, indices=[2],
                                                                                window=window)

        u = []
        v = []
        w = []

        for fft_val in fft_i2[0]:
            u.append(fft_val * vec[0])
            v.append(fft_val * vec[1])
            w.append(fft_val * vec[2])

        all_u.append(np.asarray(u))
        all_v.append(np.asarray(v))
        all_w.append(np.asarray(w))

    # new_u = []
    # new_v = []
    # new_w = []
    #
    # n_points = len(all_u[0])
    #
    # for i in range(n_points):
    #     print(i, n_points)
    #     u = []
    #     v = []
    #     w = []
    #
    #     for j in range(nchan):
    #         try:
    #             u.append(all_u[i][j])
    #             v.append(all_v[i][j])
    #             w.append(all_w[i][j])
    #         except IndexError:
    #             u.append(0)
    #             v.append(0)
    #             w.append(0)
    #
    #
    #     new_u.append(u)
    #     new_v.append(v)
    #     new_w.append(w)

    return x, y, z, all_u, all_v, all_w

def calc_fft_all(signals):
    ffts = []
    for i in range(len(signals)):
        print(i)
        signal = signals[i]
        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]
        fft_i2, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filtered_signal, indices=[2])

        ffts.append(fft_i2[0])

    return ffts


def animate_vectors():
    #fname = "many_many_successful.npz"
    fname = "sample_data38.npz"
    signals, names, time, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    names, signals = order_lists(detecs, names, signals)
    print(type(signals))

    ffts = calc_fft_all(signals)
    #print(np.shape(signals))
    #print(np.shape(ffts))
    #print(type(ffts))
    vis.helmet_animation(names, ffts, 1000, cmap="Purples", vlims=[0, 1*10**(-7)])


def angle_test():
    vect1 = [1, 0]
    vect1 = vect1 / np.linalg.norm(vect1)
    vect2 = [0, 1]
    vect2 = vect2 / np.linalg.norm(vect2)
    angles = np.linspace(0, 2 * np.pi, 5000)
    magnitude = 1

    def vect_from_ang(theta, mag):
        return [mag*np.cos(theta), mag*np.sin(theta)]

    def projection(vect1, vect2):
        dot = np.dot(vect1, vect2)
        vect1_length = np.sqrt(vect1[0]**2 + vect1[1]**2)
        return abs(dot / vect1_length)

    proj1 = []
    proj2 = []
    for angle in angles:
        vect = vect_from_ang(angle, magnitude)
        proj1.append(projection(vect1, vect))
        proj2.append(projection(vect2, vect))

    fig = plt.figure()
    plt.plot(angles, proj1, label="proj1")
    plt.plot(angles, proj2, label="proj2")
    plt.legend()
    plt.show()


def vector_closeness():
    detecs = np.load("array120_trans_newnames.npz")
    comparator = [1, 0, 0]

    for comp_name in detecs:
        print(comp_name)
        diffs = []
        comparator = detecs[comp_name][:3, 2]

        for name in detecs:
            detec = detecs[name]
            v = detec[:3, 2]
            diffs.append(sa.vect_angle(comparator, v, unit=False, perp=True))

        vis.plot_all(detecs, diffs, cmap="OrRd")


def signal_sim():
    fname = "many_successful.npz"
    all_signals, names, time, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    names, all_signals = order_lists(detecs, names, all_signals)

    comp_chan = "MEG0241"
    comp_sigs, comp_names, comp_time, n_comp = fr.get_signals(fname, channels=[comp_chan])
    comp_signal = comp_sigs[0]
    comp_filter_i = sa.filter_start(comp_signal)
    filtered_comp_sig = comp_signal[comp_filter_i:]
    comp_fft, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filtered_comp_sig, indices=[2])
    comp_fft = comp_fft[0]
    comp_v = detecs[comp_chan][:3, 2]

    all_diffs = []

    for i in range(n_chan):
        name = names[i]
        print(name)
        signal = all_signals[i]
        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]
        v = detecs[name][:3, 2]
        fft, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filtered_signal, indices=[2])
        fft = fft[0]
        diffs = sa.calc_similarity_between_signals(comp_fft, fft, comp_v, v, angle_w=0)
        all_diffs.append(diffs)
        #print(diffs)
        print()

        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        # ax1.plot(filtered_comp_sig)
        # ax1.plot(filtered_signal)
        # ax2.plot(comp_fft, label="comparator")
        # ax2.plot(fft, label=name)
        # ax2.legend()
        # ax3.plot(diffs, color="green")
        # plt.show()

    vis.helmet_animation(names, all_diffs, cmap="plasma", vlims=[0, 5])

def compare_nearby2():
    fname = "many_many_successful.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")
    window = 400

    def filter_and_fft(sig):
        filter_i = sa.filter_start(sig)
        filtered_sig = sig[filter_i:]
        x = list(range(filter_i, len(sig)))
        fft_i2, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filtered_sig, indices=[2], window=window)
        fft = fft_i2[0]
        fft_x = x[:len(x) - window]
        return filtered_sig, x, filter_i, fft, fft_x

    for i in range(n_chan):
        name = names[i]
        print(name)
        signal = signals[i]
        filtered_sig, x, filter_i, fft, fft_x = filter_and_fft(signal)
        #fft_grad = np.gradient(fft)
        og_fft_ave = np.mean(fft)

        nearby_chans = sa.find_nearby_detectors(name, detecs)
        near_sigs = fr.find_signals(nearby_chans, signals, names)
        filtered_sigs = []
        near_ffts = []
        #near_fft_grads = []
        fft_fixes = []
        near_xs = []
        near_fft_xs = []

        for sig in near_sigs:
            filt_near_sig, near_x, near_filter_i, near_fft, near_fft_x = filter_and_fft(sig)
            filtered_sigs.append(filt_near_sig)
            near_ffts.append(near_fft)
            near_fft_ave = np.mean(near_fft)
            fft_diff = og_fft_ave - near_fft_ave
            fft_fix = [x + fft_diff for x in near_fft]
            fft_fixes.append(fft_fix)
            #near_fft_grads.append(np.gradient(near_fft))
            near_xs.append(near_x)
            near_fft_xs.append(near_fft_x)

        is_similar, diff_list, new_x_list = sa.check_similarities(fft, fft_x, fft_fixes, near_fft_xs)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(near_sigs)))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
        ax1.plot(x, filtered_sig, color="black", label=name)
        ax2.plot(fft_x, fft, color="black")
        ax3.plot(fft_x, fft, color="black")

        for j in range(len(near_sigs)):
            near_name = nearby_chans[j]
            near_sig = filtered_sigs[j]
            near_x = near_xs[j]
            near_fft = near_ffts[j]
            #near_fft_grad = near_fft_grads[j]
            fft_fixed = fft_fixes[j]
            near_fft_x = near_fft_xs[j]
            near_diff = diff_list[j]
            new_x = new_x_list[j]
            similar = is_similar[j]

            diff_ave = np.mean(near_diff)
            label = near_name + " " + str(similar) + " " + str(diff_ave)

            ax1.plot(near_x, near_sig, color=colors[j], label=near_name)
            ax2.plot(near_fft_x, near_fft, color=colors[j])
            ax3.plot(near_fft_x, fft_fixed, color=colors[j])
            ax4.plot(new_x, near_diff, color=colors[j], label=label)

        print()

        ax1.set_ylabel("signal")
        ax1.legend()
        ax2.set_ylabel("fft i2")
        ax3.set_ylabel("fft i2 gradient")
        ax4.legend()
        ax4.set_ylabel("fft i2 gradient diff")

        vis.plot_all(nearby_chans, [int(x) for x in is_similar], vmin=0, vmax=1)

        plt.show()

def compare_nearby():
    fname = "many_many_successful.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")

    def plot_fft_components(og_name, og_sig, og_x, og_fft, og_v,
                            names, signals, detecs, window=400,
                            fft_ylims=[-1 * 10 ** (-7), 1 * 10 ** (-7)],
                            full_screen=False, vector_scale=1):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

        color_og = "black"
        og_fft_x = og_x[:len(og_sig) - window]
        ax1.plot(og_x, og_sig, color=color_og)
        ax2.plot(og_fft_x, og_fft, label=og_name, color=color_og)
        fracs = []

        colors = plt.cm.rainbow(np.linspace(0, 1, len(signals)))

        for i in range(len(signals)):
            signal = signals[i]
            filter_i = sa.filter_start(signal)
            x = list(range(filter_i, len(signal)))
            signal = signal[filter_i:]
            name = names[i]
            detec_matrx = detecs[name]
            r_dut = detec_matrx[:3, 3]
            v_dut = detec_matrx[:3, 2]

            #window = 400
            x_fft = x[:len(x) - window]
            fft_i2, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(signal, indices=[2], window=window)
            fft_i2 = np.gradient(fft_i2[0])
            #diffs = sa.calc_similarity_between_signals(og_fft, fft_i2, v_dut, og_v, angle_w=0)
            diffs, x_diffs = sa.calc_diff(og_fft, fft_i2, og_fft_x, x_fft)
            diff_sens = .2*10**(-10)
            diffs_under_sens = [x for x in diffs if x < diff_sens]
            frac_under = len(diffs_under_sens) / len(diffs)
            mean = np.mean(diffs)
            fracs.append(mean)
            #print(len(diffs), len_diffs)

            # print(fft_i2)

            # if len_diffs != x_fft:
            #     len_diff = len(x_fft) - len_diffs
            #     x_diffs = x_fft[:len(x_fft) - len_diff]
            # else:
            #     x_diffs = x_fft

            #x_diffs = list(range(filter_i, len_diffs))
            color = colors[i]
            print(v_dut)
            lab = name + " " + str(mean)
            print(len(x_diffs), len(diffs))
            ax1.plot(x, signal, color=color)
            ax2.plot(x_fft, fft_i2, label=name, color=color)
            ax3.plot(x_diffs, diffs, label=lab, color=color)

        title = og_name
        ax1.set_title(title)
        ax2.legend()
        ax3.legend()

        if full_screen:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()

        plt.show()

        vis.plot_all(names, fracs, vmin=0, vmax=.5*10**(-10))

    #plot_fft_components("lol", names, signals, detecs)

    #signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan)

    for i in range(n_chan):
        name = names[i]
        print(name)
        signal = signals[i]
        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]
        loc_matrx = detecs[name]
        v = loc_matrx[:3, 2]
        window = 400
        og_x = list(range(filter_i, len(signal)))
        fft, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filtered_signal, indices=[2], window=window)
        fft = np.gradient(fft[0])

        near_chans = sa.find_nearby_detectors(name, detecs)
        all_sigs = fr.find_signals(near_chans, signals, names)

        plot_fft_components(name, filtered_signal, og_x, fft, v, near_chans, all_sigs, detecs, window=window, full_screen=False)

        print()
        #plt.show()


def test_hz():
    from scipy.optimize import curve_fit
    fname = "sample_data10.npz"
    #fname = "many_many_successful2.npz"
    # channels = ["MEG0121", "MEG1114", "MEG0311", "MEG331", "MEG0334"]
    # channels = ["MEG1114"]
    # fname = "many_failed.npz"
    # channels = ["MEG0131"]
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
        # ax3.get_shared_x_axes().join(ax3, ax4)
        ax1.plot(filtered_signal, label="original signal")
        # ax1.plot(signal_x, fit(np.asarray(signal_x), *popt), label="fitted signal")
        ax1.plot(smooth_x, smooth_signal, label="smooth signal")
        ax1.plot(new_signal, label="new signal")
        ax1.legend()

        colors = plt.cm.rainbow(np.linspace(0, 1, num_indices))
        for j, c in zip(range(num_indices), colors):
            index = fft_indices[j]
            dat = i_arr[j]
            y_arr, frac_arr, hseg, smooth_frac, smooth_maxima, final_i = sa.find_default_y(dat)
            segs = sa.get_spans_from_fft(dat, hseg)
            print(segs)
            plot_spans(ax1, segs)
            # frac_grad = np.diff(frac_arr)
            ax2.plot(dat, label=str(index), color=c)
            ax2.axhspan(hseg[0], hseg[1], alpha=.5)
            ax3.plot(y_arr, frac_arr)
            ax3.plot(y_arr, smooth_frac)

            for maxi in smooth_maxima:
                if maxi == final_i:
                    color = "red"
                else:
                    color = "blue"
                ax3.axvline(y_arr[maxi], linestyle="--", color=color)

            # ax4.plot(y_arr[:len(y_arr) - 1], frac_grad, ".-", color=c)

        # ax2.set_ylim(-1 * 10**(-8), 2.2 * 10 ** (-7))
        ax2.legend()
        ax2.grid()
        plt.title(name)

        print()

        plt.show()


def secondver():
    #fname = "sample_data02.npz"
    fname = "sample_data24.npz"
    channels = ["MEG2*1"]
    signals, names, time, n_chan = fr.get_signals(fname, channels=channels)

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

    # n = data.n_channels
    # signals = sa.reorganize_signals(signals, n)
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
    # print(detecs)
    # print(detecs)
    # chan_num = len(names)
    # statuses = np.full((len(names), 2), (False, 2))
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
    # basic()
    # analysis()
    # dataload()
    # averagetest()
    # firstver()
    #secondver()
    # plottest()
    # animtest()
    # simo()
    # nearby()
    # names()
    # simulation()
    # test_uniq()
    # overlap()
    #test_hz()
    compare_nearby2()
    #animate_vectors()
    #vector_closeness()
    #angle_test()
    #signal_sim()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
