import signal_analysis as sa
import matplotlib.pyplot as plt
import numpy as np


# this file contains certain helper functions used in various places.
# not all of them are used or commented

orig_time_window = (0.210, 0.50)

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
def filter_and_smooth(signal, offset, smooth_window, smooth_only=False):
    if not smooth_only:
        filter_i = sa.filter_start(signal)
    else:
        filter_i = 0

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


def find_sigs_with_good_segs(time_to_find, t_x, names, bad_seg_list):
    good_names = []
    good_times_list = []
    good_is_list = []

    # print(bad_seg_list)

    #for i in range(len(t_x)):
    #    print(t_x[i], i)
    #print(np.where(abs(time_to_find[0] - t_x) == min(abs(time_to_find[0] - t_x))))

    start_i = np.where(abs(time_to_find[0] - t_x) == min(abs(time_to_find[0] - t_x)))[0][0]
    end_i = np.where(abs(time_to_find[1] - t_x) == min(abs(time_to_find[1] - t_x)))[0][0]
    # seg_to_find = [start_i, end_i]
    seg_to_find_list = list(range(start_i, end_i + 1))

    for i in range(len(names)):
        # print(i)
        name = names[i]
        bad_segs = bad_seg_list[i]
        # print(bad_segs)

        print(name)
        segs_of_seg_to_find = seg_to_find_list

        for bad_seg in bad_segs:
            bad_seg_temp_list = list(range(bad_seg[0], bad_seg[1] + 1))
            segs_of_seg_to_find = list(set(segs_of_seg_to_find) - set(bad_seg_temp_list))
            # print(segs_of_seg_to_find)

        segs_of_seg_to_find.sort()
        segs_of_seg_to_find_list = split_into_lists(segs_of_seg_to_find)
        # print(segs_of_seg_to_find_list)
        # print(segs_of_seg_to_find_list)

        good_times = []
        good_is = []
        for lst in segs_of_seg_to_find_list:
            seg_start_i = lst[0]
            seg_end_i = lst[-1]
            start_time = t_x[seg_start_i]
            end_time = t_x[seg_end_i]
            good_times.append([start_time, end_time])
            good_is.append([seg_start_i, seg_end_i])

        if len(good_times) != 0:
            good_names.append(name)
            good_times_list.append(good_times)
            good_is_list.append(good_is)

        # print()

    return good_names, good_times_list, good_is_list



# split a single list of integers into several lists so that each new list
# contains no gaps between each integer
def split_into_lists(original_list):
    n = len(original_list)

    if n == 0:
        return original_list

    new_lists = []
    lst = [original_list[0]]
    for i in range(1, n):
        integer = original_list[i]
        prev_int = integer - 1

        if prev_int not in lst:
            new_lists.append(lst)
            lst = [integer]
        elif i == n - 1:
            lst.append(integer)
            new_lists.append(lst)
        else:
            lst.append(integer)

    return new_lists

def i_seg_from_time_seg(time_seg, t_x):
    start_i = np.where(abs(time_seg[0] - t_x) == min(abs(time_seg[0] - t_x)))[0][0]
    end_i = np.where(abs(time_seg[1] - t_x) == min(abs(time_seg[1] - t_x)))[0][0]
    return [start_i, end_i]


def crop_signals_time(time_seg, t, signals, seg_extend):
    final_i = len(t) - 1
    i_seg = i_seg_from_time_seg(time_seg, t)
    i_seg_extend = [i_seg[0] - seg_extend, i_seg[-1] + seg_extend]

    if i_seg_extend[0] < 0:
        i_seg_extend[0] = 0

    if i_seg_extend[-1] > final_i:
        i_seg_extend[-1] = final_i

    if time_seg[0] < orig_time_window[0] or time_seg[1] > orig_time_window[1]:
        print("the window you have requested is out of bounds for the data window."
              " outputting the following window instead:", t[i_seg_extend])

    cropped_signals = []
    cropped_ix = []

    # print(i_seg_extend)

    for signal in signals:
        filter_i = sa.filter_start(signal)

        if filter_i > i_seg_extend[0]:
            start_i = filter_i
        else:
            start_i = i_seg_extend[0]

        cropped_signals.append(signal[start_i:i_seg_extend[-1]])
        cropped_ix.append(list(range(start_i, i_seg_extend[-1])))

    return cropped_signals, cropped_ix


def segs_from_i_to_time(ix_list, t_x, bad_segs):
    bad_segs_time = []
    for i in range(len(bad_segs)):
        bad_seg = bad_segs[i]
        i_x = ix_list[i]
        offset = i_x[0]

        fixed_bads = fix_segs(bad_seg, offset)

        bad_segs_time_1 = []
        for bad_seg_1 in fixed_bads:
            start_t = t_x[bad_seg_1[0]]
            end_t = t_x[bad_seg_1[-1]]
            bad_segs_time_1.append([start_t, end_t])

        bad_segs_time.append(bad_segs_time_1)

    return bad_segs_time