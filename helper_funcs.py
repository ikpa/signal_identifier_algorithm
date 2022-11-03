import signal_analysis as sa
import matplotlib.pyplot as plt
import numpy as np


# this file contains certain helper functions used in various places.
# not all of them are used or commented

# plot all segments onto a figure
def plot_spans(ax, segments, color="blue"):
    if len(segments) == 0:
        return

    for segment in segments:
        ax.axvspan(segment[0], segment[1], color=color, alpha=.5)

    return


# plot all signals as well as show data calculated by the program
def plot_in_order_ver3(signals, names, n_chan, statuses,
                       bad_seg_list, suspicious_seg_list, exec_times=[],
                       physicality=[], ylims=None):
    print_phys = not len(physicality) == 0
    for i in range(n_chan):
        name = names[i]
        signal = signals[i]
        bad = statuses[i]
        bad_segs = bad_seg_list[i]
        suspicious_segs = suspicious_seg_list[i]
        exec_time = exec_times[i] if len(exec_times) != 0 else 0

        if print_phys:
            phys = physicality[i]

            if phys == 0:
                phys_stat = ", physical"
            if phys == 1:
                phys_stat = ", unphysical"
            if phys == 2:
                phys_stat = ", physicality could not be determined"
            if phys == 3:
                phys_stat = ", not used in physicality calculation"

        else:
            phys_stat = ""

        if bad:
            status = "bad"
        else:
            status = "good"

        fig, ax = plt.subplots()
        ax.plot(signal)

        plot_spans(ax, bad_segs, color="red")
        plot_spans(ax, suspicious_segs, color="yellow")

        ax.grid()
        title = name + ": " + status + phys_stat
        ax.set_title(title)
        plt.show()


# USED FOR TESTING
# reformat list for animation function
def bad_list_for_anim(names, bads):
    bad_names = []
    for i in range(len(names)):

        if bads[i]:
            bad_names.append(names[i])

    return bad_names


# USED FOR TESTING
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


def find_min_max_ragged(arr):
    mini = None
    maxi = None
    for i in range(len(arr)):
        sub_arr = arr[i]
        for val in sub_arr:
            if maxi is None or val > maxi:
                maxi = val

            if mini is None or val < mini:
                mini = val

    return mini, maxi


def get_single_point(signals, i, n=1):
    points = []

    for signal in signals:
        point = signal[i]
        for j in range(n):
            points.append(point)

    return points


def exclude_from_lists(i, lists):
    new_lists = []

    print(len(lists[0]))

    for lis in lists:
        excl_val = lis[i]
        new_list = [x for x in lis if not np.array_equal(excl_val, x)]
        new_lists.append(new_list)

    return new_lists


# filter the beginning spike from a signal and smooth it
def filter_and_smooth(signal, offset, smooth_window):
    filter_i = sa.filter_start(signal)
    filtered_signal = signal[filter_i:]
    x = list(range(filter_i, len(signal)))

    smooth_signal = sa.smooth(filtered_signal, window_len=smooth_window)
    smooth_x = [x - offset + filter_i for x in list(range(len(smooth_signal)))]
    new_smooth = []
    for i in range(len(filtered_signal)):
        new_smooth.append(smooth_signal[i + offset])

    return filtered_signal, x, smooth_signal, smooth_x, new_smooth


def fix_segs(segs, offset):
    new_segs = []
    for seg in segs:
        new_segs.append([seg[0] + offset, seg[1] + offset])

    return new_segs
