import signal_analysis as sa
import matplotlib.pyplot as plt
import numpy as np


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


def filter_and_fft(sig, window):
    filter_i = sa.filter_start(sig)
    filtered_sig = sig[filter_i:]
    x = list(range(filter_i, len(sig)))
    fft_i2, smooth_signal, smooth_x, detrended_signal = sa.calc_fft_indices(filtered_sig, indices=[2], window=window)
    fft = fft_i2[0]
    fft_x = x[:len(x) - window]
    return filtered_sig, x, filter_i, fft, fft_x
