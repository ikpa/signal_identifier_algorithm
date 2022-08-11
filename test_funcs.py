import helmet_vis as vis
import numpy as np
import helper_funcs as hf
import file_reader as fr
import signal_analysis as sa
import matplotlib.pyplot as plt
import signal_generator as sg

datadir = "example_data_for_patrik/"

def animate_vectors():
    fname = "many_successful.npz"
    #fname = "sample_data38.npz"
    signals, names, time, n_chan = fr.get_signals(fname)
    detecs = np.load("array120_trans_newnames.npz")
    names, signals = hf.order_lists(detecs, names, signals)
    print(type(signals))

    ffts = hf.calc_fft_all(signals)
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
        #fft_grad = np.gradient(fft)
        og_fft_ave = np.mean(fft)

        nearby_chans = sa.find_nearby_detectors(name, detecs)
        near_sigs = fr.find_signals(nearby_chans, signals, names)
        filtered_sigs = []
        near_ffts = []
        near_fft_grads = []
        fft_fixes = []
        correlations = []
        near_xs = []
        near_fft_xs = []

        #new_x_list = []

        for sig in near_sigs:
            filt_near_sig, near_x, near_filter_i, near_fft, near_fft_x = hf.filter_and_fft(sig, window)
            filtered_sigs.append(filt_near_sig)
            near_ffts.append(near_fft)
            near_fft_ave = np.mean(near_fft)
            fft_diff = og_fft_ave - near_fft_ave
            fft_fix = [x + fft_diff for x in near_fft]
            fft_fixes.append(fft_fix)
            #near_fft_grads.append(np.gradient(near_fft))
            #corr = correlate(fft, fft_fix)
            #correlations.append(corr)
            near_xs.append(near_x)
            near_fft_xs.append(near_fft_x)

            new_fft_og, new_near_fft, new_x = sa.crop_signals(fft, near_fft, x, near_fft_x)
            #print(len(new_fft_og), len(new_near_fft), len(new_x))
            #new_x_list.append(new_x)
            #corr = correlate(fft, fft_fix, "full")
            corr = sm.tsa.stattools.ccf(new_fft_og, new_near_fft)
            correlations.append(corr)

            #is_similar, diff_list, new_x_list = sa.check_similarities(fft, fft_x, fft_fixes, near_fft_xs)

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
            #near_fft_grad = near_fft_grads[j]
            fft_fixed = fft_fixes[j]
            near_fft_x = near_fft_xs[j]
            correl = correlations[j]
            #new_x = new_x_list[j]
            #print(correl)
            #near_diff = diff_list[j]
            #new_x = new_x_list[j]
            #similar = is_similar[j]

            #diff_ave = np.mean(near_diff)
            #label = near_name + " " + str(similar) + " " + str(diff_ave)

            ax1.plot(near_x, near_sig, color=colors[j], label=near_name)
            ax2.plot(near_fft_x, near_fft, color=colors[j])
            ax3.plot(near_fft_x, fft_fixed, color=colors[j])
            ax4.plot(correl, color=colors[j], label=near_name)

        print()

        ax1.set_ylabel("signal")
        ax1.legend()
        ax2.set_ylabel("fft i2")
        #ax3.set_ylabel("fft i2 gradient")
        ax3.legend()
        #ax4.set_ylabel("fft i2 gradient diff")

        #vis.plot_all(nearby_chans, [int(x) for x in is_similar], vmin=0, vmax=1)

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
    vis.helmet_animation(detecs, signals, 1000)


def nearby():
    detecs = np.load("array120_trans_newnames.npz")
    dut = "MEG1011"
    nears = sa.find_nearby_detectors(dut, detecs)
    print(nears)
    matrixes, names, bads = hf.regex_filter(detecs)
    bads.append(nears)
    print(bads)

    vis.plot_all(names, np.full(np.shape(names), 1), nears)

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

    vis.helmet_animation(names, signals, frames=1000, bads=bad_list)

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

def detrend_grad():
    fname = "many_many_successful.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")
    smooth_window = 401
    offset = int(smooth_window / 2)

    def filter_and_smooth(signal):
        filter_i = sa.filter_start(signal)
        filtered_signal = signal[filter_i:]
        x = list(range(filter_i, len(signal)))

        smooth_signal = sa.smooth(filtered_signal, window_len=smooth_window)
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

        near_names = sa.find_nearby_detectors(name, detecs)
        near_sigs = fr.find_signals(near_names, signals, names)

        for j in range(len(near_names)):
            near_name = near_names[j]
            near_sig = near_sigs[j]

            near_filtered_signal, near_x, near_smooth_signal, near_smooth_x, near_crop_smooth = filter_and_smooth(near_sig)
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
    smooth_window = 401
    offset = int(smooth_window / 2)

    fname = "sample_data37.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")

    for k in range(n_chan):
        # comp_detec = "MEG2411"
        #comp_detec = "MEG0711"
        comp_detec = names[k]
        #comp_sig = fr.find_signals([comp_detec], signals, names)[0]
        comp_sig = signals[k]
        comp_v = detecs[comp_detec][:3, 2]
        comp_r = detecs[comp_detec][:3, 3]

        nearby_names = sa.find_nearby_detectors(comp_detec, detecs)
        near_sigs = fr.find_signals(nearby_names, signals, names)

        near_vs = []
        near_rs = []

        for name in nearby_names:
            near_vs.append(detecs[name][:3, 2])
            near_rs.append(detecs[name][:3, 3])

        nearby_names.append(comp_detec)
        near_sigs.append(comp_sig)
        near_vs.append(comp_v)
        near_rs.append(comp_r)

        signal_statuses, bad_segment_list, suspicious_segment_list, exec_times = sa.analyse_all_neo(near_sigs,
                                                                                                    nearby_names,
                                                                                                    len(near_sigs))

        #hf.plot_in_order_ver3(near_sigs, nearby_names, len(near_sigs), signal_statuses, bad_segment_list,
        #                      suspicious_segment_list, exec_times)

        filtered_sigs = []
        xs = []
        calc_names = []
        calc_vs = []
        calc_rs = []

        exclude_chans = ["MEG0724", "MEG0634", "MEG0744"]
        exclude_chans = []

        for i in range(len(near_sigs)):
            nam = nearby_names[i]

            if nam in exclude_chans:
                print("excluding " + nam)
                continue

            if len(bad_segment_list[i]) != 0:
                print("bad segments found, skipping " + nam)
                continue

            signal = near_sigs[i]

            filtered_signal, x, smooth_signal, smooth_x, new_smooth = hf.filter_and_smooth(signal, offset, smooth_window)
            filtered_sigs.append(np.gradient(new_smooth))
            xs.append(x)
            calc_names.append(nam)
            calc_vs.append(near_vs[i])
            calc_rs.append(near_rs[i])

        magnus, mag_is, cropped_sigs, new_x = sa.calc_magn_field_from_signals(filtered_sigs, xs, calc_vs, ave_window=1)

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

            diffs, diff_x = sa.calc_diff(reconst_sig, cropped_sigs[i], mag_is, new_x)
            # print(diffs)
            all_diffs.append(diffs)
            diff_xs.append(diff_x)
            diff_aves.append(np.mean(diffs))

        ave_of_aves = np.mean(diff_aves)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

        colors = plt.cm.rainbow(np.linspace(0, 1, len(calc_names)))
        for i in range(len(cropped_sigs)):
            color = colors[i]
            plot_name = calc_names[i]
            ax1.plot(new_x, cropped_sigs[i], color=color, label=plot_name)
            ax2.plot(mag_is, reconst_sigs[i], color=color, label=plot_name)
            ax3.plot(diff_xs[i], all_diffs[i], color=color, label=diff_aves[i])

        ax1.set_title(ave_of_aves)
        ax2.legend()
        ax3.legend()
        plt.show()


def test_excluder():
    smooth_window = 401
    offset = int(smooth_window / 2)

    fname = "sample_data37.npz"
    signals, names, time, n_chan = fr.get_signals(fname)

    detecs = np.load("array120_trans_newnames.npz")

    comp_detec = "MEG1631"
    #comp_detec = "MEG0711"
    comp_sig = fr.find_signals([comp_detec], signals, names)[0]
    comp_v = detecs[comp_detec][:3, 2]
    comp_r = detecs[comp_detec][:3, 3]

    nearby_names = sa.find_nearby_detectors(comp_detec, detecs)
    near_sigs = fr.find_signals(nearby_names, signals, names)

    near_vs = []
    near_rs = []

    for name in nearby_names:
        near_vs.append(detecs[name][:3, 2])
        near_rs.append(detecs[name][:3, 3])

    nearby_names.append(comp_detec)
    near_sigs.append(comp_sig)
    near_vs.append(comp_v)
    near_rs.append(comp_r)

    smooth_sigs = []
    xs = []

    for i in range(len(near_sigs)):
        signal = near_sigs[i]
        filtered_signal, x, smooth_signal, smooth_x, new_smooth = hf.filter_and_smooth(signal, offset, smooth_window)
        smooth_sigs.append(np.gradient(new_smooth))
        xs.append(x)

    exclude_chans = sa.filter_unphysical_sigs(smooth_sigs, nearby_names, xs, near_vs)
    #print(exclude_chans)
    exclude_is = [i for i in range(len(nearby_names)) if nearby_names[i] in exclude_chans]

    good_sigs = []
    good_names = []
    good_xs = []
    good_vs = []

    for i in range(len(nearby_names)):
        if i in exclude_is:
            continue

        good_sigs.append(smooth_sigs[i])
        good_names.append(nearby_names[i])
        good_xs.append(xs[i])
        good_vs.append(near_vs[i])

    print(good_names)

    ave_of_aves, aves, diffs, rec_sigs = sa.rec_and_diff(smooth_sigs, xs, near_vs)
    #print(diffs)
    good_ave_of_aves, good_aves, good_diffs, good_rec_sigs = sa.rec_and_diff(good_sigs, good_xs, good_vs)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(nearby_names)))
    good_colors = []
    for i in range(len(colors)):
        if i in exclude_is:
            continue
        good_colors.append(colors[i])

    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    for i in range(len(smooth_sigs)):
        color = colors[i]
        plot_name = nearby_names[i]
        ax1.plot(smooth_sigs[i], color=color, label=plot_name)
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

