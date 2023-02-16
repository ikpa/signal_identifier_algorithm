import fdfs as sa
import matplotlib.pyplot as plt
import numpy as np


"""this file contains certain helper functions used in various places.
not all of them are used or commented"""

orig_time_window = (0.210, 0.50)

def plot_spans(ax, segments, color="blue"):
    """plot all segments onto a figure"""
    if len(segments) == 0:
        return

    for segment in segments:
        ax.axvspan(segment[0], segment[1], color=color, alpha=.5)

    return


def seg_to_time(x, segs):
    new_segs = []
    for seg in segs:
        new_segs.append(x[seg])

    return new_segs

def plot_in_order_ver3(signals, names, n_chan, statuses,
                       bad_seg_list, suspicious_seg_list, exec_times=[],
                       physicality=[], time_x=None, ylims=None, showtitle=True):
    """# plot all signals as well as show data calculated by the program"""
    print_phys = not len(physicality) == 0

    #plt.rcParams.update({'font.size': 42})
    for i in range(n_chan):
        name = names[i]
        # print(name)
        signal = signals[i]
        bad = statuses[i]
        bad_segs = bad_seg_list[i]
        suspicious_segs = suspicious_seg_list[i]
        exec_time = exec_times[i] if len(exec_times) != 0 else 0

        if time_x is not None:
            bad_segs = seg_to_time(time_x, bad_segs)
            suspicious_segs = seg_to_time(time_x, suspicious_segs)

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
            status = "bad segments present"
        else:
            status = "no bad segments found"

        fig, ax = plt.subplots(figsize=(12,10))
        plt.tight_layout(rect=(0.02, 0.02, 0.98, 0.98))
        linewidth = 4

        if time_x is None:
            ax.plot(signal, linewidth=linewidth)
        else:
            ax.plot(time_x, signal, linewidth=linewidth)

        plot_spans(ax, bad_segs, color="red")
        plot_spans(ax, suspicious_segs, color="yellow")

        ax.grid()
        ax.set_ylabel("Magnetic Field [T]")
        ax.set_xlabel("Time [s]")

        if showtitle:
            title = name + ": " + status + phys_stat
            ax.set_title(title)

        plt.show()
        print()

def bad_list_for_anim(names, bads):
    """USED FOR TESTING
    reformat list for animation function"""
    bad_names = []
    for i in range(len(names)):

        if bads[i]:
            bad_names.append(names[i])

    return bad_names

def order_lists(pos_list, dat_names, signals):
    """USED FOR TESTING"""
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

    #print(len(lists[0]))

    for lis in lists:
        excl_val = lis[i]
        new_list = [x for x in lis if not np.array_equal(excl_val, x)]
        new_lists.append(new_list)

    return new_lists

def filter_and_smooth(signal, offset, smooth_window, smooth_only=False):
    """filter the beginning spike from a signal and smooth it"""
    if not smooth_only:
        filter_i = sa.filter_start(signal)
    else:
        filter_i = 0

    filtered_signal = signal[filter_i:]
    x = list(range(filter_i, len(signal)))

    smooth_signal = smooth(filtered_signal, window_len=smooth_window)
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


def split_into_lists(original_list):
    """split a single list of integers into several lists so that each new list
    contains no gaps between each integer"""
    n = len(original_list)

    if n == 0:
        return original_list

    original_list.sort()
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

    return cropped_signals, cropped_ix, i_seg


def segs_from_i_to_time(ix_list, t_x, bad_segs):
    ix_is_list = not isinstance(ix_list[0], int)
    bad_segs_time = []
    for i in range(len(bad_segs)):
        bad_seg = bad_segs[i]
        if ix_is_list:
            i_x = ix_list[i]
        else:
            i_x = ix_list
        offset = i_x[0]

        fixed_bads = fix_segs(bad_seg, offset)

        bad_segs_time_1 = []
        for bad_seg_1 in fixed_bads:
            start_t = t_x[bad_seg_1[0]]
            end_t = t_x[bad_seg_1[-1]]
            bad_segs_time_1.append([start_t, end_t])

        bad_segs_time.append(bad_segs_time_1)

    return bad_segs_time


def find_good_segs(i_x_target, bad_seg_list, i_x_tot):
    ix_set = set(list(range(i_x_target[0] - i_x_tot[0], i_x_target[1] - i_x_tot[0] + 1)))
    good_seg_is = []

    for i in range(len(bad_seg_list)):
        bad_segs = bad_seg_list[i]

        bad_i_set = set()
        for bad_seg in bad_segs:
            bad_list = list(range(bad_seg[0], bad_seg[-1] + 1))
            bad_i_set.update(bad_list)

        good_list_all = list(ix_set - bad_i_set)

        split_good_list = split_into_lists(good_list_all)

        good_is = []
        for good_list in split_good_list:
            good_is.append([good_list[0], good_list[-1]])

        good_seg_is.append(good_is)

        if len(good_is) > 1:
            print(good_list_all)
            print(good_is)

    return good_seg_is


def find_signals(channels, data_arr, names):
    """finds and returns data corresponding to channel names. data_arr and
    names must be ordered and have the same length"""
    indices = []

    for channel in channels:
        i = names.index(channel)
        indices.append(i)

    signals_to_return = []

    for index in indices:
        signals_to_return.append(data_arr[index])

    return signals_to_return


def reorganize_signals(signals, n):
    """reformats signals so that a single array contains one signal instead of
    one array containing one point in time"""
    new_signals = []
    for i in range(n):
        signal = signals[:, i]
        new_signals.append(signal)

    return new_signals

def filter_and_smooth_and_gradient_all(signals, offset, smooth_window, smooth_only=False):
    filt_sigs = []
    xs = []

    for signal in signals:
        filtered_signal, x, smooth_signal, smooth_x, new_smooth = filter_and_smooth(signal, offset, smooth_window,
                                                                                    smooth_only=smooth_only)

        filt_sigs.append(np.gradient(filtered_signal))
        xs.append(x)

    return filt_sigs, xs

def list_good_sigs(names, signals, bad_seg_list, sus_seg_list):
    good_names = []
    good_signals = []
    good_sus = []
    good_bad = []

    for i in range(len(names)):
        bad_segs = bad_seg_list[i]

        if len(bad_segs) != 0:
            good_names.append(names[i])
            good_signals.append(signals[i])
            good_sus.append(sus_seg_list[i])
            good_bad.append([])


    return good_names, good_signals, good_sus, good_bad


def smooth(x, window_len=21, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def crop_all_sigs(signals, xs, bad_segs):
    """takes several signals and crops them on the x axis so that all signals
    are of the same length. if bad segments are present, only the part
    before these segments are included. x-values must be in indices"""
    highest_min_x = 0
    lowest_max_x = 10 ** 100

    # find the min and max x values that are shared by all signals
    for x in xs:
        min_x = np.amin(x)
        max_x = np.amax(x)

        if min_x > highest_min_x:
            highest_min_x = min_x

        if max_x < lowest_max_x:
            lowest_max_x = max_x

    # remove all x values appearing in or after all bad segments
    for seg_list in bad_segs:
        for seg in seg_list:
            if lowest_max_x > seg[0]:
                lowest_max_x = seg[0]

    new_x = list(range(highest_min_x, lowest_max_x))
    new_signals = []

    # get parts of all signals that appear within the new x values
    for i in range(len(signals)):
        signal = signals[i]
        x = xs[i]
        max_i = x.index(lowest_max_x)
        min_i = x.index(highest_min_x)
        new_signals.append(signal[min_i:max_i])

    return new_signals, new_x


def averaged_signal(signal, ave_window, x=[], mode=0):
    """calculate rolling operation to signal. returns a signal that is
    len(signal)/ave_window data points long.
    modes:
    0 = average
    1 = rms
    2 = sdev."""
    new_sig = []
    new_x = []

    start_i = 0
    end_i = ave_window
    max_i = len(signal) - 1

    cont = True
    while cont:
        seg = signal[start_i:end_i]
        if mode == 0:
            ave = np.mean(seg)

        if mode == 1:
            ave = np.sqrt(np.mean([x ** 2 for x in seg]))

        if mode == 2:
            ave = np.std(seg)

        new_sig.append(ave)

        if len(x) != 0:
            new_x.append(int(np.mean([x[start_i], x[end_i]])))

        start_i = end_i
        end_i = end_i + ave_window

        if end_i > max_i:
            end_i = max_i

        if start_i >= max_i:
            cont = False

    if len(x) != 0:
        return new_sig, new_x

    return new_sig


def calc_diff(signal1, signal2, x1, x2):
    """calculate absolute difference between points in two different signals.
    inputs may have different x-values with different spacings, as long as
    there is some overlap. x-values must be in indices"""
    new_x = []
    diffs = []
    for i in range(len(signal1)):
        new_x.append(x1[i])
        point1 = signal1[i]
        point2 = signal2[i]
        diffs.append(abs(point1 - point2))

    return diffs, new_x

def find_nearby_detectors(d_name, detectors, good_names, r_sens=0.06):
    """find detectors within a radius r_sens from a given detector. channels
    not in good_names are excluded."""
    dut = detectors[d_name]
    r_dut = dut[:3, 3]

    nears = []

    for name in detectors:
        if name == d_name or name not in good_names:
            continue

        detector = detectors[name]
        r = detector[:3, 3]
        delta_r = np.sqrt((r_dut[0] - r[0]) ** 2 +
                          (r_dut[1] - r[1]) ** 2 +
                          (r_dut[2] - r[2]) ** 2)

        if delta_r < r_sens:
            nears.append(name)

    return nears


def length_of_segments(segments):
    tot_len = 0
    for segment in segments:
        length = segment[1] - segment[0]
        tot_len += length

    return tot_len