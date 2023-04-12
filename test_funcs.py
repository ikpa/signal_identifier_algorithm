import time

#import helmet_vis as vis
import numpy as np

import fdfs
import helper_funcs
import helper_funcs as hf
import file_handler as fr
import pca
import fdfs as sa
import matplotlib.pyplot as plt
import signal_generator as sg
import helmet_vis as vis

# from sklearn import preprocessing

datadir = "example_data_for_patrik/"

brokens = ["MEG0111", "MEG0221", "MEG0234", "MEG0241", "MEG1244", "MEG2131", "MEG2244"]


# various test functions. not commented

def animate_vectors():
    fname = "many_successful.npz"
    # fname = "sample_data38.npz"
    signals, names, time, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    names, signals = hf.order_lists(detecs, names, signals)
    print(type(signals))

    ffts = hf.calc_fft_all(signals)
    # print(np.shape(signals))
    # print(np.shape(ffts))
    # print(type(ffts))
    # vis.helmet_animation(names, ffts, 1000, cmap="Purples", vlims=[0, 1 * 10 ** (-7)])


def angle_test():
    vect1 = [1, 0]
    vect1 = vect1 / np.linalg.norm(vect1)
    vect2 = [0, 1]
    vect2 = vect2 / np.linalg.norm(vect2)
    angles = np.linspace(0, 2 * np.pi, 5000)
    magnitude = 1

    def vect_from_ang(theta, mag):
        return [mag * np.cos(theta), mag * np.sin(theta)]

    def projection(vect1, vect2):
        dot = np.dot(vect1, vect2)
        vect1_length = np.sqrt(vect1[0] ** 2 + vect1[1] ** 2)
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


def signal_sim():
    fname = "many_successful.npz"
    all_signals, names, time, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    names, all_signals = hf.order_lists(detecs, names, all_signals)

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
        # print(diffs)
        print()

        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        # ax1.plot(filtered_comp_sig)
        # ax1.plot(filtered_signal)
        # ax2.plot(comp_fft, label="comparator")
        # ax2.plot(fft, label=name)
        # ax2.legend()
        # ax3.plot(diffs, color="green")
        # plt.show()

    # vis.helmet_animation(names, all_diffs, cmap="plasma", vlims=[0, 5])


def compare_nearby2():
    from scipy.signal import correlate
    import statsmodels.api as sm
    fname = "many_many_successful.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")
    window = 400

    for i in range(n_chan):
        name = names[i]
        print(name)
        signal = signals[i]
        filtered_sig, x, filter_i, fft, fft_x = hf.filter_and_fft(signal, window)
        # fft_grad = np.gradient(fft)
        og_fft_ave = np.mean(fft)

        nearby_chans = helper_funcs.find_nearby_detectors(name, detecs)
        near_sigs = fr.find_signals(nearby_chans, signals, names)
        filtered_sigs = []
        near_ffts = []
        near_fft_grads = []
        fft_fixes = []
        correlations = []
        near_xs = []
        near_fft_xs = []

        # new_x_list = []

        for sig in near_sigs:
            filt_near_sig, near_x, near_filter_i, near_fft, near_fft_x = hf.filter_and_fft(sig, window)
            filtered_sigs.append(filt_near_sig)
            near_ffts.append(near_fft)
            near_fft_ave = np.mean(near_fft)
            fft_diff = og_fft_ave - near_fft_ave
            fft_fix = [x + fft_diff for x in near_fft]
            fft_fixes.append(fft_fix)
            # near_fft_grads.append(np.gradient(near_fft))
            # corr = correlate(fft, fft_fix)
            # correlations.append(corr)
            near_xs.append(near_x)
            near_fft_xs.append(near_fft_x)

            new_fft_og, new_near_fft, new_x = sa.crop_signals(fft, near_fft, x, near_fft_x)
            # print(len(new_fft_og), len(new_near_fft), len(new_x))
            # new_x_list.append(new_x)
            # corr = correlate(fft, fft_fix, "full")
            corr = sm.tsa.stattools.ccf(new_fft_og, new_near_fft)
            correlations.append(corr)

            # is_similar, diff_list, new_x_list = sa.check_similarities(fft, fft_x, fft_fixes, near_fft_xs)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(near_sigs)))

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
        ax1.plot(x, filtered_sig, color="black", label=name)
        ax2.plot(fft_x, fft, color="black")
        ax3.plot(fft_x, fft, color="black")

        for j in range(len(near_sigs)):
            near_name = nearby_chans[j]
            near_sig = filtered_sigs[j]
            near_x = near_xs[j]
            near_fft = near_ffts[j]
            # near_fft_grad = near_fft_grads[j]
            fft_fixed = fft_fixes[j]
            near_fft_x = near_fft_xs[j]
            correl = correlations[j]
            # new_x = new_x_list[j]
            # print(correl)
            # near_diff = diff_list[j]
            # new_x = new_x_list[j]
            # similar = is_similar[j]

            # diff_ave = np.mean(near_diff)
            # label = near_name + " " + str(similar) + " " + str(diff_ave)

            ax1.plot(near_x, near_sig, color=colors[j], label=near_name)
            ax2.plot(near_fft_x, near_fft, color=colors[j])
            ax3.plot(near_fft_x, fft_fixed, color=colors[j])
            ax4.plot(correl, color=colors[j], label=near_name)

        print()

        ax1.set_ylabel("signal")
        ax1.legend()
        ax2.set_ylabel("fft i2")
        # ax3.set_ylabel("fft i2 gradient")
        ax3.legend()
        # ax4.set_ylabel("fft i2 gradient diff")

        # vis.plot_all(nearby_chans, [int(x) for x in is_similar], vmin=0, vmax=1)

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

            # window = 400
            x_fft = x[:len(x) - window]
            fft_i2, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(signal, indices=[2], window=window)
            fft_i2 = np.gradient(fft_i2[0])
            # diffs = sa.calc_similarity_between_signals(og_fft, fft_i2, v_dut, og_v, angle_w=0)
            diffs, x_diffs = helper_funcs.calc_diff(og_fft, fft_i2, og_fft_x, x_fft)
            diff_sens = .2 * 10 ** (-10)
            diffs_under_sens = [x for x in diffs if x < diff_sens]
            frac_under = len(diffs_under_sens) / len(diffs)
            mean = np.mean(diffs)
            fracs.append(mean)
            # print(len(diffs), len_diffs)

            # print(fft_i2)

            # if len_diffs != x_fft:
            #     len_diff = len(x_fft) - len_diffs
            #     x_diffs = x_fft[:len(x_fft) - len_diff]
            # else:
            #     x_diffs = x_fft

            # x_diffs = list(range(filter_i, len_diffs))
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

        # vis.plot_all(names, fracs, vmin=0, vmax=.5 * 10 ** (-10))

    # plot_fft_components("lol", names, signals, detecs)

    # signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan)

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
        fft, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filtered_signal, indices=[2],
                                                                             window=window)
        fft = np.gradient(fft[0])

        near_chans = helper_funcs.find_nearby_detectors(name, detecs)
        all_sigs = fr.find_signals(near_chans, signals, names)

        plot_fft_components(name, filtered_signal, og_x, fft, v, near_chans, all_sigs, detecs, window=window,
                            full_screen=False)

        print()
        # plt.show()


def test_hz():
    from scipy.optimize import curve_fit
    fname = "sample_data10.npz"
    # fname = "many_many_successful2.npz"
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
            hf.plot_spans(ax1, segs)
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
    # vis.helmet_animation(detecs, signals, 1000)


def nearby():
    detecs = np.load("array120_trans_newnames.npz")
    dut = "MEG1011"
    nears = helper_funcs.find_nearby_detectors(dut, detecs)
    print(nears)
    matrixes, names, bads = hf.regex_filter(detecs)
    bads.append(nears)
    print(bads)

    # vis.plot_all(names, np.full(np.shape(names), 1), nears)


def plottest():
    fname = datadir + "many_successful.npz"
    data = fr.load_all(fname).subpool(["MEG*1", "MEG*4"]).clip((0.210, 0.50))
    unorganized_signals = data.data
    names = data.names
    n = data.n_channels
    signals = sa.reorganize_signals(unorganized_signals, n)
    detecs = np.load("array120_trans_newnames.npz")

    names, signals = hf.order_lists(detecs, names, signals)

    # n = data.n_channels
    # signals = sa.reorganize_signals(signals, n)
    statuses, bad_segs, sus_segs, exec_times = sa.analyse_all_neo(signals, names, n)
    bad_list = hf.bad_list_for_anim(names, statuses)

    # vis.helmet_animation(names, signals, frames=1000, bads=bad_list)


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

        where_repeat, conf = sa.flat_filter(signal)
        segments += where_repeat
        seg_confidences += conf

        where_repeat, conf = sa.spike_filter_neo(signal, filter_i)
        segments += where_repeat
        seg_confidences += conf

        print(segments)
        # print(seg_confidences)
        seg_i_list.append(sa.combine_segments(segments))
        # confidences.append(seg_confidences)
        print(segments)
        print()

    hf.plot_in_order_neo(signals, names, n_chan, bads, seg_is=seg_i_list, confidence_list=confidences)


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

        hf.plot_spans(ax2, segments)

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


def detrend_grad():
    fname = datadir + "many_many_successful.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")
    smooth_window = 401
    offset = int(smooth_window / 2)

    def filter_and_smooth(signal):
        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]
        x = list(range(filter_i, len(signal)))

        smooth_signal = helper_funcs.smooth(filtered_signal, window_len=smooth_window)
        smooth_x = [x - offset + filter_i for x in list(range(len(smooth_signal)))]
        new_smooth = []
        for i in range(len(filtered_signal)):
            new_smooth.append(smooth_signal[i + offset])

        crop_smooth = [a - b for a, b in zip(signal, new_smooth)]

        return filtered_signal, x, smooth_signal, smooth_x, new_smooth

    for i in range(n_chan):
        name = names[i]
        print(name)
        signal = signals[i]

        filtered_signal, x, smooth_signal, smooth_x, crop_smooth = filter_and_smooth(signal)
        grad = np.gradient(crop_smooth)

        xs = []
        filtered_sigs = []
        smooths = []
        smooth_xs = []
        smooth_grads = []

        near_names = helper_funcs.find_nearby_detectors(name, detecs)
        near_sigs = fr.find_signals(near_names, signals, names)

        for j in range(len(near_names)):
            near_name = near_names[j]
            near_sig = near_sigs[j]

            near_filtered_signal, near_x, near_smooth_signal, near_smooth_x, near_crop_smooth = filter_and_smooth(
                near_sig)
            near_grad = np.gradient(near_crop_smooth)
            smooth_grads.append(near_grad)
            filtered_sigs.append(near_filtered_signal)
            xs.append(near_x)
            smooths.append(near_crop_smooth)
            smooth_xs.append(near_smooth_x)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(x, filtered_signal, color="black", label=name)
        ax2.plot(x, crop_smooth, color="black", label=name)
        ax3.plot(x, grad, color="black", label=name)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(near_sigs)))

        for j in range(len(near_sigs)):
            color = colors[j]
            plot_name = near_names[j]
            sig = filtered_sigs[j]
            plot_x = xs[j]
            smooth_plot = smooths[j]
            smooth_x_plot = smooth_xs[j]
            plot_grad = smooth_grads[j]
            print(len(smooth_x_plot), len(plot_x))
            ax1.plot(plot_x, sig, color=color, label=plot_name)
            ax2.plot(plot_x, smooth_plot, color=color, label=plot_name)
            ax3.plot(plot_x, plot_grad, color=color, label=plot_name)

        ax3.legend()
        plt.show()


def test_magn():
    smooth_window = 201
    offset = int(smooth_window / 2)

    fname = "sample_data37.npz"
    signals, names, t, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")

    def seg_lens(sig, segs):
        length = 0
        for seg in segs:
            length += seg[1] - seg[0]

        return length / len(sig)

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names,
                                                                                                len(signals))

    for k in range(n_chan):
        # comp_detec = "MEG2411"
        # comp_detec = "MEG0711"
        comp_detec = names[k]
        # comp_sig = fr.find_signals([comp_detec], signals, names)[0]
        comp_sig = signals[k]
        comp_v = detecs[comp_detec][:3, 2]
        comp_r = detecs[comp_detec][:3, 3]

        nearby_names = helper_funcs.find_nearby_detectors(comp_detec, detecs)
        nearby_names.append(comp_detec)
        # near_segs = fr.find_signals(nearby_names, bad_segment_list, names)

        new_near = []
        # new_segs = []
        for nam in nearby_names:
            index = names.index(nam)
            bad = seg_lens(signals[index], bad_segment_list[index]) > .5

            if bad:
                print("excluding " + nam + " from calculation")
                continue

            new_near.append(nam)

        nearby_names = new_near

        near_sigs = fr.find_signals(nearby_names, signals, names)

        near_vs = []
        near_rs = []

        for name in nearby_names:
            near_vs.append(detecs[name][:3, 2])
            near_rs.append(detecs[name][:3, 3])

        # hf.plot_in_order_ver3(near_sigs, nearby_names, len(near_sigs), signal_statuses, bad_segment_list,
        #                      suspicious_segment_list, exec_times)

        filtered_sigs = []
        xs = []
        calc_names = []
        calc_vs = []
        calc_rs = []

        for i in range(len(near_sigs)):
            nam = nearby_names[i]

            signal = near_sigs[i]

            filtered_signal, x, smooth_signal, smooth_x, new_smooth = hf.filter_and_smooth(signal, offset,
                                                                                           smooth_window)
            filtered_sigs.append(np.gradient(new_smooth))
            xs.append(x)
            calc_names.append(nam)
            calc_vs.append(near_vs[i])
            calc_rs.append(near_rs[i])

        magnus, mag_is, cropped_sigs, new_x = pca.calc_magn_field_from_signals(filtered_sigs, xs, calc_vs, ave_window=1)

        reconst_sigs = []
        all_diffs = []
        diff_xs = []
        diff_aves = []

        for i in range(len(calc_rs)):
            r = calc_rs[i]
            v = calc_vs[i]
            reconst_sig = []

            for j in range(len(magnus)):
                magn = magnus[j]
                reconst_sig.append(np.dot(magn, v))

            reconst_sigs.append(reconst_sig)

            diffs, diff_x = helper_funcs.calc_diff(reconst_sig, cropped_sigs[i], mag_is, new_x)
            # print(diffs)
            all_diffs.append(diffs)
            diff_xs.append(diff_x)
            diff_aves.append(np.mean(diffs))

        # ave_of_aves = np.mean(diff_aves)
        # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        #
        # colors = plt.cm.rainbow(np.linspace(0, 1, len(calc_names)))
        # for i in range(len(cropped_sigs)):
        #     color = colors[i]
        #     plot_name = calc_names[i]
        #     ax1.plot(new_x, cropped_sigs[i], color=color, label=plot_name)
        #     ax2.plot(mag_is, reconst_sigs[i], color=color, label=plot_name)
        #     ax3.plot(diff_xs[i], all_diffs[i], color=color, label=diff_aves[i])
        #
        # ax1.set_title(ave_of_aves)
        # ax2.legend()
        # ax3.legend()
        # plt.show()

        # print(exclude_chans)
        exclude_chans = []
        exclude_is = [i for i in range(len(nearby_names)) if nearby_names[i] in exclude_chans]

        good_sigs = []
        good_names = []
        good_xs = []
        good_vs = []
        # print(cropped_sigs)
        # print(exclude_is)
        print(len(cropped_sigs))
        print(len(nearby_names))
        for i in range(len(nearby_names)):
            if i in exclude_is:
                continue

            # print(i)
            good_sigs.append(cropped_sigs[i])
            good_names.append(nearby_names[i])
            good_xs.append(xs[i])
            good_vs.append(near_vs[i])

        print(good_names)

        ave_of_aves, aves, diffs, rec_sigs = pca.rec_and_diff(cropped_sigs, xs, near_vs)
        # print(diffs)
        good_ave_of_aves, good_aves, good_diffs, good_rec_sigs = pca.rec_and_diff(good_sigs, good_xs, good_vs)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(nearby_names)))
        good_colors = []
        for i in range(len(colors)):
            if i in exclude_is:
                continue
            good_colors.append(colors[i])

        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

        for i in range(len(cropped_sigs)):
            color = colors[i]
            plot_name = nearby_names[i]
            ax1.plot(cropped_sigs[i], color=color, label=plot_name)
            ax2.plot(rec_sigs[i], color=color, label=plot_name)
            ax3.plot(diffs[i], color=color, label=aves[i])

        ax1.set_title(ave_of_aves)
        ax2.legend()
        ax3.legend()

        fig2, (ax11, ax22, ax33) = plt.subplots(3, 1, sharex=True)

        for i in range(len(good_sigs)):
            color = good_colors[i]
            plot_name = good_names[i]
            ax11.plot(good_sigs[i], color=color, label=plot_name)
            ax22.plot(good_rec_sigs[i], color=color, label=plot_name)
            ax33.plot(good_diffs[i], color=color, label=good_aves[i])

        ax11.set_title(good_ave_of_aves)
        ax22.legend()
        ax33.legend()

        plt.show()


def test_magn2():
    smooth_window = 401
    offset = int(smooth_window / 2)
    printer = fr.Printer("print")
    #plt.rcParams.update({'font.size': 42})
    crop = False

    fname = datadir + "many_many_successful.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)

    if crop:
        time_window = [0.4, 0.45]
        precropped_signals, precropped_ix = hf.crop_signals_time(time_window, timex, signals, 200)

    detecs = np.load("array120_trans_newnames.npz")

    ave_window = 100
    ave_sens = 10 ** (-13)

    def seg_lens(sig, segs):
        length = 0
        for seg in segs:
            length += seg[1] - seg[0]

        return length / len(sig)

    if crop:
        signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(precropped_signals, names,
                                                                                                len(signals), filter_beginning=False)
    else:
        signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals,
                                                                                                    names,
                                                                                                    len(signals), printer,
                                                                                                    filter_beginning=True)

    for i in range(len(signals)):
        comp_detec = names[i]
        nearby_names = helper_funcs.find_nearby_detectors(comp_detec, detecs, names)
        nearby_names.append(comp_detec)

        new_names = []
        for name in nearby_names:
            index = names.index(name)
            bad_segs = bad_segment_list[index]
            bad = len(bad_segs) != 0

            if bad:
                print("excluding " + name)
                continue

            new_names.append(name)

        #print(len(new_names))

        if len(new_names) == 0:
            print()
            continue

        near_vs = []
        near_rs = []

        for name in new_names:
            near_vs.append(detecs[name][:3, 2])
            near_rs.append(detecs[name][:3, 3])

        # nearby_names.append(comp_detec)
        if crop:
            near_sigs = hf.find_signals(new_names, precropped_signals, names)
        else:
            near_sigs = hf.find_signals(new_names, signals, names)

        near_sus = hf.find_signals(new_names, suspicious_segment_list, names)
        near_bads = hf.find_signals(new_names, bad_segment_list, names)

        smooth_sigs = []
        detrended_sigs = []
        xs = []

        for k in range(len(near_sigs)):
            signal = near_sigs[k]
            # filter_i = sa.filter_start(signal)
            # filt_sig = signal[filter_i:]
            filtered_signal, x, smooth_signal, smooth_x, new_smooth = hf.filter_and_smooth(signal, offset,
                                                                                           smooth_window, smooth_only=crop)

            smooth_sigs.append(np.gradient(new_smooth))
            xs.append(x)

            if False:
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 10))
                plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))
                linewidth = 4
                ax1.plot(timex[x], filtered_signal, linewidth=linewidth)
                ax2.plot(timex[x], new_smooth, linewidth=linewidth)
                ax3.plot(timex[x], np.gradient(new_smooth), linewidth=linewidth)
                ax3.set_xlabel("Time [s]")
                ax1.set_ylabel("Magn. F. [T]")
                ax2.set_ylabel("Magn. F. [T]")
                ax3.set_ylabel("Magn. F. \nDeriv. [T/s]")
                ax1.grid()
                ax2.grid()
                ax3.grid()

            # i_arr, ix, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filt_sig, [2])
            # smooth_sigs.append(i_arr[0])
            # xs.append(ix)
            # detrended_sigs.append(detrended_signal)

            # smooth_sigs.append(smooth_signal)
            # xs.append(x)

        # for signal in smooth_sigs:
        # print(len(signal))

        cropped_signals, o_new_x = helper_funcs.crop_all_sigs(smooth_sigs, xs, [])

        # print(len(new_x))

        # print(len(cropped_signals))
        ave_of_aves, aves, all_diffs, rec_sigs, mag_is, new_cropped_signals, crop_x = pca.rec_and_diff(cropped_signals,
                                                                                                       [o_new_x],
                                                                                                       near_vs, printer,
                                                                                                       ave_window=ave_window)

        if False:
            fig, ax = plt.subplots(figsize=(12, 10))
            plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))
            linewidth = 4
            ax.plot(timex[xs[0]], smooth_sigs[0], linewidth=linewidth, label="Preprocessed signal")
            ax.plot(timex[mag_is], rec_sigs[0], linewidth=linewidth, label="Reconstructed signal")
            ax.set_ylabel("Magn. F. \nDeriv. [T/s]")
            ax.set_xlabel("Time [s]")
            ax.legend()
            ax.grid()

        excludes, new_x, ave_diffs, rel_diffs = pca.filter_unphysical_sigs(smooth_sigs, new_names, xs, near_vs,
                                                                           near_bads, near_sus, printer,
                                                                           ave_window=ave_window, ave_sens=ave_sens)

        # print(o_new_x, new_x)

        if cropped_signals is None or rec_sigs is None:
            print()
            continue

        good_sigs = []
        good_names = []
        good_vs = []
        good_xs = []

        for k in range(len(smooth_sigs)):
            nam = new_names[k]

            if nam in excludes:
                continue

            good_sigs.append(cropped_signals[k])
            good_names.append(nam)
            good_vs.append(near_vs[k])
            # good_xs.append(xs[k])
            good_xs.append(o_new_x)

        good_ave_of_aves, good_aves, good_all_diffs, good_rec_sigs, good_mag_is, good_cropped_signals, good_crop_x = pca.rec_and_diff(
            good_sigs, [o_new_x], good_vs, printer, ave_window=ave_window)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(new_names)))
        good_colors = []
        for k in range(len(colors)):
            nam = new_names[k]
            if nam in excludes:
                continue
            good_colors.append(colors[k])

        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        #fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        #plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))
        linewidth = 1.5

        for k in range(len(cropped_signals)):
            color = colors[k]
            plot_name = new_names[k]
            ax1.plot(timex[xs[k]], smooth_sigs[k], color=color, label=plot_name, linewidth=linewidth)
            ax2.plot(timex[mag_is], rec_sigs[k], color=color, label=plot_name, linewidth=linewidth)
            ax3.plot(timex[mag_is], all_diffs[k], color=color, label=aves[k])

        #ax1.set_title(ave_of_aves)
        #ax2.legend()
        ax1.set_ylabel("Magn. F. \nDeriv. [T/s]")
        ax2.set_ylabel("Magn. F. \nDeriv. [T/s]")
        ax2.set_xlabel("Time [s]")
        ax1.grid()
        ax2.grid()
        #ax3.legend()

        fig2, (ax11, ax22, ax33) = plt.subplots(3, 1, sharex=True)
        #fig12, (ax11, ax22) = plt.subplots(2, 1, sharex=True, figsize=(12, 10))
        #plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))

        for k in range(len(good_sigs)):
            color = good_colors[k]
            plot_name = good_names[k]
            ax11.plot(timex[good_xs[k]], good_sigs[k], color=color, label=plot_name, linewidth=linewidth)
            ax22.plot(timex[good_mag_is], good_rec_sigs[k], color=color, label=plot_name, linewidth=linewidth)
            ax33.plot(timex[good_mag_is], good_all_diffs[k], color=color, label=good_aves[k])

        #ax11.set_title(good_ave_of_aves)
        #ax22.legend()
        #ax33.legend()
        ax11.set_ylabel("Magn. F. \nDeriv. [T/s]")
        ax22.set_ylabel("Magn. F. \nDeriv. [T/s]")
        ax22.set_xlabel("Time [s]")
        ax11.grid()
        ax22.grid()

        if len(new_x) != 0:
            print(new_x[0], new_x[-1])
            #hf.plot_spans(ax11, [[new_x[0], new_x[-1]]])
            #hf.plot_spans(ax1, [[new_x[0], new_x[-1]]])
        else:
            print("no span")

        fig3, ax = plt.subplots()
        for k in range(len(near_sigs)):
            plot_name = new_names[k]
            color = colors[k]
            ax.plot(near_sigs[k], label=plot_name, color=color)

        ax.legend()

        print()

        plt.show()


def test_new_excluder():
    smooth_window = 401
    offset = int(smooth_window / 2)

    printer = fr.Printer("print")
    crop = False

    fname = datadir + "sample_data34.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)

    if crop:
        time_window = [0.22, 0.25]
        signals, ix, i_seg = hf.crop_signals_time(time_window, timex, signals, 200)
        signals, ix = helper_funcs.crop_all_sigs(signals, ix, [])

    detecs = np.load("array120_trans_newnames.npz")

    # times_excluded = np.zeros(n_chan)
    # times_in_calc = np.zeros(n_chan)

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan, printer,
                                                                                                filter_beginning=(not crop))

    start_time = time.time()
    all_diffs, all_rel_diffs, chan_dict = pca.check_all_phys(signals, detecs, names, n_chan, bad_segment_list, suspicious_segment_list,
                                                             printer, ave_window=100, ave_sens=10 ** (-13), smooth_only=crop)
    end_time = time.time()

    status, confidence = pca.analyse_phys_dat_alt(all_diffs, names, all_rel_diffs, chan_dict)

    ex_time = (end_time - start_time) / 60
    print("execution time:", ex_time, "mins")

    titles = []

    for i in range(len(chan_dict)):
        stat = status[i]

        if stat == 0:
            st_string = "physical"
        elif stat == 1:
            st_string = "unphysical"
        elif stat == 2:
            st_string = "undetermined"
        elif stat == 3:
            st_string = "unused"
        # num_exc = times_excluded[i]
        # num_tot = times_in_calc[i]
        nam = names[i]

        diffs = all_diffs[nam]
        rel_diffs = all_rel_diffs[nam]
        chan_dat = chan_dict[nam]

        num_tot = len(chan_dat)
        num_exc = len([x for x in chan_dat if chan_dat[x] == 1])
        # print(num_tot, num_exc)
        num_exc = np.float64(num_exc)
        num_tot = np.float64(num_tot)

        tit = nam + " " + str(num_exc) + "/" + str(num_tot) + "=" + \
              str(num_exc / num_tot) + ": " + st_string + ", " + str(confidence[i])

        titles.append(tit)

        print(nam, num_exc, num_tot, np.float64(num_exc / num_tot), st_string, confidence[i], np.mean(rel_diffs))

    for i in range(len(chan_dict)):
        nam = names[i]
        chan_dat = chan_dict[nam]
        num_tot = len(chan_dat)
        num_exc = len([x for x in chan_dat if chan_dat[x] == 1])
        num_exc = np.float64(num_exc)
        num_tot = np.float64(num_tot)
        signal = signals[i]
        segs = bad_segment_list[i]

        nearby_names = hf.find_nearby_detectors(nam, detecs, names)
        sigs = hf.find_signals(nearby_names, signals, names)

        # if num_exc == 0:
        #    continue

        colors = plt.cm.rainbow(np.linspace(0, 1, len(nearby_names) + 1))
        fig, ax = plt.subplots()
        ax.plot(signal, label=nam, color=colors[0])
        hf.plot_spans(ax, segs)
        ax.set_title(titles[i])

        for j in range(len(nearby_names)):
            name = nearby_names[j]
            sig = sigs[j]
            ax.plot(sig, label=name, color=colors[j + 1])

        ax.legend()

        plt.show()


def test_excluder():
    smooth_window = 401
    offset = int(smooth_window / 2)

    fname = datadir + "sample_data30.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")

    times_excluded = np.zeros(n_chan)
    times_in_calc = np.zeros(n_chan)

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
                                                                                                badness_sensitivity=.5)

    start_time = time.time()

    for k in range(n_chan):
        # comp_detec = "MEG2221"
        # comp_detec = "MEG0711"
        comp_detec = names[k]
        print(comp_detec)
        # comp_sig = signals[k]
        # comp_sig = fr.find_signals([comp_detec], signals, names)[0]
        # comp_v = detecs[comp_detec][:3, 2]
        # comp_r = detecs[comp_detec][:3, 3]

        nearby_names = helper_funcs.find_nearby_detectors(comp_detec, detecs)
        nearby_names.append(comp_detec)
        # new_near = nearby_names

        new_near = []
        # new_segs = []
        for nam in nearby_names:
            index = names.index(nam)
            bad = signal_statuses[index]

            if bad:
                print("excluding " + nam + " from calculation")
                continue

            new_near.append(nam)
            # new_segs.append(bad_segment_list[index])

        near_vs = []
        near_rs = []

        for name in new_near:
            near_vs.append(detecs[name][:3, 2])
            near_rs.append(detecs[name][:3, 3])

        # nearby_names.append(comp_detec)
        near_sigs = fr.find_signals(new_near, signals, names)
        cluster_bad_segs = fr.find_signals(new_near, bad_segment_list, names)
        # near_sigs.append(comp_sig)
        # near_vs.append(comp_v)
        # near_rs.append(comp_r)

        smooth_sigs = []
        xs = []

        for i in range(len(near_sigs)):
            signal = near_sigs[i]
            filtered_signal, x, smooth_signal, smooth_x, new_smooth = hf.filter_and_smooth(signal, offset,
                                                                                           smooth_window)
            smooth_sigs.append(np.gradient(new_smooth))
            xs.append(x)

        exclude_chans, new_x, ave_diffs, rel_diffs = pca.filter_unphysical_sigs(smooth_sigs, new_near, xs, near_vs, cluster_bad_segs,
                                                                                ave_window=1)

        if len(new_x) != 0:
            for nam in new_near:
                index = names.index(nam)
                times_in_calc[index] += 1

        if len(new_x) > 1:
            print("analysed segment between", new_x[0], new_x[len(new_x) - 1])

        print("excluded", exclude_chans)
        for chan in exclude_chans:
            exclude_i = names.index(chan)
            times_excluded[exclude_i] += 1
            ex = times_excluded[exclude_i]
            tot = times_in_calc[exclude_i]
            print(chan, ex, tot, ex / tot)

        print()

        # # print(exclude_chans)
        # exclude_is = [i for i in range(len(nearby_names)) if nearby_names[i] in exclude_chans]
        #
        # good_sigs = []
        # good_names = []
        # good_xs = []
        # good_vs = []
        #
        # for i in range(len(nearby_names)):
        #     if i in exclude_is:
        #         continue
        #
        #     good_sigs.append(smooth_sigs[i])
        #     good_names.append(nearby_names[i])
        #     good_xs.append(xs[i])
        #     good_vs.append(near_vs[i])

    end_time = time.time()
    ex_time_sec = end_time - start_time
    ex_time_min = ex_time_sec / 60
    print("execution time " + str(ex_time_min) + " mins")

    for i in range(len(times_excluded)):
        if len(bad_segment_list[i]) != 0:
            filtered = "filtered"
        else:
            filtered = "not filtered"
        num_exc = times_excluded[i]
        num_tot = times_in_calc[i]
        nam = names[i]

        print(nam, num_exc, num_tot, num_exc / num_tot, filtered)

    for i in range(len(times_excluded)):
        num_exc = times_excluded[i]
        num_tot = times_in_calc[i]
        signal = signals[i]
        nam = names[i]
        segs = bad_segment_list[i]

        # if num_exc == 0:
        #    continue

        fig, ax = plt.subplots()
        ax.plot(signal)
        hf.plot_spans(ax, segs)
        ax.set_title(nam + " " + str(num_exc) + " " + str(num_tot) + " " + str(num_exc / num_tot))
        plt.show()

    # print(good_names)
    #
    # ave_of_aves, aves, diffs, rec_sigs = sa.rec_and_diff(smooth_sigs, xs, near_vs)
    # #print(diffs)
    # good_ave_of_aves, good_aves, good_diffs, good_rec_sigs = sa.rec_and_diff(good_sigs, good_xs, good_vs)
    #
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(nearby_names)))
    # good_colors = []
    # for i in range(len(colors)):
    #     if i in exclude_is:
    #         continue
    #     good_colors.append(colors[i])
    #
    # fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    #
    # for i in range(len(smooth_sigs)):
    #     color = colors[i]
    #     plot_name = nearby_names[i]
    #     ax1.plot(smooth_sigs[i], color=color, label=plot_name)
    #     ax2.plot(rec_sigs[i], color=color, label=plot_name)
    #     ax3.plot(diffs[i], color=color, label=aves[i])
    #
    # ax1.set_title(ave_of_aves)
    # ax2.legend()
    # ax3.legend()
    #
    # fig2, (ax11, ax22, ax33) = plt.subplots(3, 1, sharex=True)
    #
    # for i in range(len(good_sigs)):
    #     color = good_colors[i]
    #     plot_name = good_names[i]
    #     ax11.plot(good_sigs[i], color=color, label=plot_name)
    #     ax22.plot(good_rec_sigs[i], color=color, label=plot_name)
    #     ax33.plot(good_diffs[i], color=color, label=good_aves[i])
    #
    # ax11.set_title(good_ave_of_aves)
    # ax22.legend()
    # ax33.legend()
    #
    # plt.show()


def test_gradient():
    smooth_window = 401
    offset = int(smooth_window / 2)

    fname = datadir + "many_successful.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)

    for i in range(n_chan):
        signal = signals[i]
        sig_len = len(signal)
        name = names[i]
        print(name)
        filter_i = sa.filter_start(signal)
        filt_sig = signal[filter_i:]
        filt_x = list(range(filter_i, sig_len))

        smooth_signal = helper_funcs.smooth(filt_sig, window_len=smooth_window)
        smooth_x = [x - offset + filter_i for x in list(range(len(smooth_signal)))]

        new_smooth = []
        for j in range(filter_i, sig_len):
            new_smooth.append(smooth_signal[j + offset - filter_i])

        grad = np.gradient(new_smooth)
        max_grad = np.amax(grad)
        min_grad = abs(np.amin(grad))

        norm = max(max_grad, min_grad)

        if norm != 0:
            norm_grad = grad / norm
        else:
            norm_grad = grad

        sens = .05

        low_vals = []
        low_x = []
        for j in range(len(grad)):
            val = norm_grad[j]
            x_val = filt_x[j]

            if -sens < val < sens:
                low_vals.append(val)
                low_x.append(x_val)

        low_x_lists = helper_funcs.split_into_lists(low_x)

        segs = []
        unfilt_segs = []
        for j in range(len(low_x_lists)):
            lst = low_x_lists[j]
            seg_start = lst[0]
            seg_end = lst[len(lst) - 1]

            unfilt_segs.append([seg_start, seg_end])

            if j != 0:
                prev_lst = low_x_lists[j - 1]
                prev_end = prev_lst[-1]

                if seg_start - prev_end < 300 and len(segs) != 0:
                    betw_vals = norm_grad[prev_end: seg_start]
                    hi_betw_vals = [x for x in betw_vals if abs(x) > .5]

                    if len(hi_betw_vals) != 0:
                        segs.append([seg_start, seg_end])
                        continue

                    # print(j - 1)
                    segs[-1][1] = seg_end
                    continue

            segs.append([seg_start, seg_end])

        filt_segs = []
        for seg in segs:
            seg_len = seg[1] - seg[0]

            if seg_len < 150:
                continue

            filt_segs.append(seg)

        means = []
        stds = []
        for seg in filt_segs:
            vals = norm_grad[seg[0]: seg[1]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        print(means)
        print(stds)
        print()

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        # ax1.plot(filt_x, filt_sig, label="unsmooth")
        ax1.plot(filt_x, new_smooth, label="smooth")
        ax1.legend()

        ax2.plot(filt_x, norm_grad)
        ax2.plot(low_x, low_vals, ".")
        hf.plot_spans(ax2, filt_segs)
        # hf.plot_spans(ax2, unfilt_segs, color="red")

        ax3.plot(filt_x, np.gradient(norm_grad))

        plt.show()



def test_fft_full():
    fname = datadir + "many_many_successful2.npz"
    channels = ["MEG2*1"]
    signals, names, timex, n_chan = fr.get_signals(fname)

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
                                                                                                badness_sensitivity=.5)

    hf.plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segment_list, suspicious_segment_list)


def test_fft():
    # SUS ONES: sd37: 2214
    # sd 32 is weird
    fname = datadir + "sample_data37.npz"
    #channels = ["MEG0624"]
    channels = ["MEG*1", "MEG*4"]
    signals, names, timex, n_chan = fr.get_signals(fname, channels=channels)
    printer = fr.Printer("print")

    detecs = np.load("array120_trans_newnames.npz")

    #time_window = [0.4, 0.5]
    #cropped_signals, cropped_ix = hf.crop_signals_time(time_window, timex, signals, 200)

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
                                                                                                printer,
                                                                                                filters=["uniq", "flat", "spike"],
                                                                                                filter_beginning=True)
    lol = False
    #plt.rcParams.update({'font.size': 41})
    for i in range(n_chan):
        name = names[i]
        print(name)
        signal = signals[i]
        # signal = signals[i]
        bad_segs = bad_segment_list[i]
        filter_i = sa.filter_start(signal)

        filter_i_list = sa.filter_start(signal, debug=True)[0]

        #normal_sig = signal[filter_i:]
        #normal_x = list(range(filter_i, len(signal)))
        full_x = list(range(len(signal)))

        indices = [2]
        fft_window = 400

        nu_i_x, nu_i_arr, u_filter_i_i, u_i_arr_ave, u_i_arr_sdev, u_cut_grad, u_grad_ave, u_grad_x, status, sus_score, rms_x, fft_sdev, error_start, sdev_thresh, sdev_span, detrended_sig = sa.fft_filter(
            signal, filter_i, bad_segs, printer, indices=indices, fft_window=fft_window, debug=True, goertzel=False, lol=lol)

        if status == 0:
            s = "GOOD"

        if status == 1:
            s = "BAD"

        if status == 2:
            s = "UNDETERMINED"

        print("SIGNAL FFT STATUS: " + s, status)
        print("SUS SCORE:", sus_score)

        if nu_i_x is None:
            print("skip")
            print()
            continue


        #u_filter_i_i = u_filter_i_i + filter_i[0]
        #u_grad_x = [x + filter_i for x in u_grad_x]

        #nu_i_x = [x for x in nu_i_x]
        #u_filter_i_i = u_filter_i_i
        #u_grad_x = [x for x in u_grad_x]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

        #fig, (ax1) = plt.subplots(1, 1, figsize=(12, 10))
        #plt.tight_layout(rect=(0.045, 0.02, 0.99, 0.98))
        linewidth = 1.5

        #error_start = None
        #status = -1
        ax1.plot(timex[full_x], signal, linewidth=linewidth)
        ax1.axvline(timex[filter_i], linestyle="--", color="blue")
        #[ax1.axvline(timex[x], linestyle="--", color="black", linewidth=0.5) for x in filter_i_list]

        #filter_i = filter_i[0]
        nu_i_x = [x + filter_i for x in nu_i_x]
        #print(nu_i_arr)
        #print("lol")
        offset = int(len(nu_i_arr) * 0.0875)
        filter_i_i_list, largest = sa.filter_start(nu_i_arr, offset=offset, max_rel=.175, debug=True)
        figlol, (axlol1, axlol2) = plt.subplots(2, 1, sharex=True)
        axlol1.plot(np.gradient(filter_i_i_list))
        axlol2.plot(largest)
        axlol2.set_ylim(0, 10*10**(-10))
        axlol2.grid()
        axlol1.grid()
        ax3.plot(timex[nu_i_x], np.gradient(nu_i_arr))
        #print(filter_i_i_list)
        filter_i_i_list = [x + filter_i for x in filter_i_i_list]
        #u_filter_i_i = u_filter_i_i + filter_i
        # [ax2.axvline(timex[x], linestyle="--", color="black", linewidth=0.5) for x in filter_i_i_list]
        #u_filter_i_i = u_filter_i_i[0]

        #ax2.plot(timex[normal_x], detrended_sig, linewidth=linewidth)
        #ax2.set_yticklabels([])
        ax2.grid()
        #ax1.plot(full_x, signal)
        hf.plot_spans(ax1, hf.seg_to_time(timex, bad_segs), color="darkred")
        #hf.plot_spans(ax1, bad_segs)

        if error_start is not None:
            hf.plot_spans(ax1, hf.seg_to_time(timex, [[error_start, len(signal) - 1]]), color="red")
            #hf.plot_spans(ax1, [[error_start, len(signal) - 1]], color="red")
        elif status == 1:
            hf.plot_spans(ax1, hf.seg_to_time(timex, [[u_filter_i_i, len(signal) - 1]]), color="red")
            #hf.plot_spans(ax1, [[filter_i, len(signal) - 1]], color="red")
        elif status == 2:
            hf.plot_spans(ax1, hf.seg_to_time(timex, [[u_filter_i_i, len(signal) - 1]]), color="yellow")
            #hf.plot_spans(ax1, [[filter_i, len(signal) - 1]], color="yellow")

        #ax1.grid()
        ax1.set_ylabel("Magnetic Field [T]")
        #ax2.set_ylabel("Mag. F. [T]")

        if nu_i_x is not None:
             ax2.plot(timex[nu_i_x], nu_i_arr, linewidth=linewidth)
             #ax2.plot(timex[nu_i_x], nu_i_arr)
             pass

        ax2.set_ylim(-0.1 * 10 ** (-7), 1.2 * 10 ** (-7))
        ax3.grid()
        ax1.set_xlabel("Time [s]")
        #ax3.set_ylabel("FT Amp.")

        ax1.grid()

        # ax2.legend()

        ax2.axvline(timex[u_filter_i_i], linestyle="--", color="blue", linewidth=0.5)
        #ax3.set_ylim(-.25 * 10 ** (-9), .25 * 10 ** (-9))

        #ax3.plot(u_grad_x, u_cut_grad, label="orig")

        # fig2, ax11 = plt.subplots()

        # rms_x, fft_sdev, error_start, sdev_thresh, sdev_span = sa.find_saturation_point_from_fft(nu_i_x, nu_i_arr, u_filter_i_i, fft_window)

        if error_start is not None:
            #pass
            ax1.axvline(timex[error_start], linestyle="--", color="red")

        if sdev_span is not None:

            #pass
            hf.plot_spans(ax4, hf.seg_to_time(timex, [sdev_span + filter_i]))
            rms_x = [x + filter_i for x in rms_x]
            ax4.plot(timex[rms_x], fft_sdev, label="sdev")
            ax4.axhline(sdev_thresh, linestyle="--", color="black")
        #
        # # ax4.plot(rms_x, fft_rms, label="rms")
        # ax4.legend()
        # ax4.set_ylim(0, 2 * 10 ** (-9))

        #ax1.legend()
        #ax2.set_ylim(0, 10 ** (-7))

        # fig, (ax1) = plt.subplots(1, 1, sharex=True)
        #
        # ax4.set_ylim(-.5 * 10 ** (-9), .5 * 10 ** (-9))
        # ax3.axhline(thresh, linestyle="--", color="black")
        # ax3.axhline(-thresh, linestyle="--", color="black")
        # ax3.legend()

        plt.show()

        print()


def show():
    channels = ["MEG0511", "MEG0541"]
    fname = datadir + "many_successful.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)

    #print(1/0.0001)

    #signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
    #                                                                                            badness_sensitivity=.5,
    #filter_beginning=True)

    plt.rcParams.update({'font.size': 42})
    for i in range(n_chan):
        signal = signals[i]
        name = names[i]
        #bad_segs = bad_segment_list[i]
        #sus_segs = suspicious_segment_list[i]

        print(name)
        print()

        # plt.plot(timex, signal)
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        linewidth = 4
        plt.plot(timex, signal, linewidth=linewidth)
        #plt.title(name)
        plt.ylabel("Magnetic Field [T]")
        plt.xlabel("Time [s]")
        plt.grid()

        #hf.plot_spans(ax, bad_segs, color="red")
        #hf.plot_spans(ax, sus_segs, color="yellow")

        plt.show()

def test_crop():
    fname = datadir + "sample_data40.npz"
    channels = ["MEG2*1"]
    signals, names, timex, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
                                                                                                badness_sensitivity=.5,
                                                                                             filter_beginning=False)

    hf.plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segment_list, suspicious_segment_list)


def test_ffft():
    fname = datadir + "sample_data40.npz"
    # channels = ["MEG1041"]
    signals, names, timex, n_chan = fr.get_signals(fname)

    printer = fr.Printer("print")
    detecs = np.load("array120_trans_newnames.npz")

    indices = [2]

    for i in range(n_chan):
        signal = signals[i]
        name = names[i]
        print(name)
        filter_i = sa.filter_start(signal)
        signal = signal[filter_i:]

        start_time = time.time()
        i_arr, nu_x, smooth_signal, smooth_x, filtered_signal = sa.calc_fft_indices(signal, printer, indices=indices)
        end_time = time.time()
        orig_time = end_time - start_time

        start_time = time.time()
        new_i_arr, nu_x, smooth_signal, smooth_x, filtered_signal = sa.calc_fft_index_fast(signal, printer)
        end_time = time.time()
        new_time = end_time - start_time

        plt.plot(i_arr[0], label="orig, " + str(orig_time))
        plt.plot(new_i_arr, label="new, " + str(new_time))
        plt.legend()
        plt.show()



def show_pca():
    # mms 1241 -> bad exc
    skp = "MEG1044"
    fname = datadir + "sample_data02.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    printer = fr.Printer("print")
    skip = False

    start_t = time.time()
    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan, printer)
    all_diffs, all_rel_diffs, chan_dict = pca.check_all_phys(signals, detecs, names, n_chan, bad_segment_list, suspicious_segment_list,
                                                             printer, ave_window=100, ave_sens=5 * 10 ** (-13))
    phys_stat, phys_conf = pca.analyse_phys_dat_alt(all_diffs, names, all_rel_diffs, chan_dict)

    end_t = time.time()

    print(end_t - start_t)

    # plt.rcParams.update({'font.size': 21})
    remove_bads = True

    for i in range(n_chan):
        name = names[i]
        print(name)
        if skip and not name == skp:
            continue
        near_names = helper_funcs.find_nearby_detectors(name, detecs, names)
        near_names.append(name)
        sigs = hf.find_signals(near_names, signals, names)
        phys_stats = hf.find_signals(near_names, phys_stat, names)
        fracs = hf.find_signals(near_names, phys_conf, names)

        if remove_bads:
            bads = hf.find_signals(near_names, bad_segment_list, names)
            index_list = []
            for ind in range(len(bads)):
                b = bads[ind]
                if len(b) != 0:
                    index_list.append(ind)
                    print("removing " + near_names[ind])

            if len(index_list) > 0:
                for indy in sorted(index_list, reverse=True):
                    del sigs[indy]
                    del near_names[indy]
                    del phys_stats[indy]
                    del fracs[indy]

        if len(sigs) == 0:
            continue

        phys_stats_string = []
        for stat in phys_stats:
            if stat == 0:
                strng = "consistent"

            if stat == 1:
                strng = "inconsistent"

            if stat == 2:
                strng = "undetermined"

            if stat == 3:
                strng = "not used"

            phys_stats_string.append(strng)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(near_names)))
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))

        for j in range(len(near_names)):
            nam = near_names[j]
            sig = sigs[j]
            status = phys_stats_string[j]
            frac = fracs[j]

            lbl = nam + ": " + status
            if not status == "not used":
                lbl += ", exc. frac.:" + str(round(frac, 2))


            if nam == name:
                c = "black"
            else:
                c = colors[j]

            ax.plot(timex, sig, color=c, label=lbl, linewidth=4)

        ax.grid()
        ax.legend()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Magnetic Field [T]")
        plt.show()

        print()


def test_fft_emergency():
    fname = datadir + "many_many_successful.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    printer = fr.Printer("print")



    for i in range(n_chan):
        name = names[i]
        signal = signals[i]
        filter_i = sa.filter_start(signal)
        filt_sig = signal[filter_i:]
        orig_x = list(range(len(signal)))

        indices = list(range(1, 11))
        i_arr, nu_x, smooth_signal, smooth_x, filtered_signal = sa.calc_fft_indices(filt_sig, printer, indices)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(orig_x, signal)
        for j in range(len(i_arr)):
            ind = str(indices[j])
            fft = i_arr[j]

            ax2.plot(nu_x, fft, label=ind)

        ax1.set_title(name)
        ax2.legend()
        ax2.set_ylim(-0.1 * 10 ** (-7), 1.2 * 10 ** (-7))

        plt.show()

def show_helmet():
    fname = datadir + "many_many_successful.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    names, signals = hf.order_lists(detecs, names, signals)

    new_dat = []

    for i in range(len(names)):
        name = names[i]

        if name[-1] == "4":
            new_dat.append(1)
        else:
            new_dat.append(0)

    vis.plot_all(names, new_dat)


def test_flat_new():
    fname = datadir + "sample_data23.npz"
    signals, names, timex, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    printer = fr.Printer("print")

    signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(signals, names, n_chan,
                                                                                                printer)


    for i in range(n_chan):
        signal = signals[i]
        name = names[i]
        bad_segs = bad_segment_list[i]
        sus_segs = suspicious_segment_list[i]

        print(name)
        #segs, scores = fdfs.flat_filter(signal, printer)

        fig, ax = plt.subplots()
        plt.plot(signal)
        hf.plot_spans(ax, bad_segs, color="red")
        hf.plot_spans(ax, sus_segs, color="yellow")
        plt.title(name)

        print()
        plt.show()
