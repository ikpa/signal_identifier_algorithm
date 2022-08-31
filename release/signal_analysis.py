import math

import numpy as np
import time
import helper_funcs as hf
from operator import itemgetter


# all sensitivity and weight values in this file have been determined
# experimentally and changing them will affect the accuracy of the program
# significantly

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


# calculates the magnetic field vector from a set of signal magnitudes (point)
# and detectors direction vectors (a) using pseudoinverse
def magn_from_point(a, point):
    a_pinv = np.linalg.pinv(a)
    magn = a_pinv.dot(point)
    return magn


# takes several signals and crops them on the x axis so that all signals
# are of the same length. if bad segments are present, only the part
# before these segments are included. x-values must be in indices
def crop_all_sigs(signals, xs, bad_segs):
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

    # remove all x values appearing after all bad segments
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


# calculates a magnetic field vector as a function of time from a set of signals
# and their vectors. the magnetic fields are calculated from averaged
# signals. averaging window can be changed using ave_window
def calc_magn_field_from_signals(signals, xs, vectors, ave_window=400):
    # crop signals if needed
    if len(xs) == 1:
        cropped_signals = signals
        new_x = xs[0]
    else:
        cropped_signals, new_x = crop_all_sigs(signals, xs, [])

    # if there are less than 3 signals, the calculation is not performed
    if len(signals) < 3:
        print("not enough signals to calculate magnetic field vector")
        return [], [], cropped_signals, new_x

    magn_vectors = []
    mag_is = []

    len_sig = len(new_x)
    max_i = len_sig - 1
    cont = True
    start_i = 0
    end_i = ave_window

    while cont:
        # calculate rolling average
        aves = []
        for i in range(len(signals)):
            signal = signals[i]
            segment = signal[start_i:end_i]
            aves.append(np.mean(segment))

        # calculate magnetic field vector
        magn = magn_from_point(vectors, aves)
        magn_vectors.append(magn)
        mag_is.append(int(np.mean([start_i, end_i])))

        # calculate next averaging window
        start_i = end_i
        end_i = end_i + ave_window

        if end_i > max_i:
            end_i = max_i

        if start_i >= max_i:
            cont = False

    # calculate the x-values for the signals (in indices)
    mag_i_offset = new_x[0] - mag_is[0]
    mag_is = [x + mag_i_offset for x in mag_is]

    return magn_vectors, mag_is, cropped_signals, new_x


# reconstruct a signal using a magnetic vector (as a function of time) mag
# and a detector direction vector
def reconstruct(mag, v):
    rec_sig = []
    for mag_point in mag:
        rec_sig.append(np.dot(mag_point, v))

    return rec_sig


# using pseudoinverse, reconstruct a set of signals and calculate their difference
# to the original signals. returns average total difference for each signal,
# average total difference for all signals in total, the reconstructed and cropped
# signals and their x values. increasing ave_window decreases calculation time
# and accuracy
def rec_and_diff(signals, xs, vs, ave_window=1):
    if len(signals) == 0:
        return None, None, None, None, None, None, None

    # calculate magnetic field vector
    magn_vectors, mag_is, cropped_signals, new_x = calc_magn_field_from_signals(signals, xs, vs,
                                                                                ave_window=ave_window)

    if len(magn_vectors) == 0:
        return None, None, None, None, None, None, None

    rec_sigs = []
    aves = []
    all_diffs = []
    for i in range(len(cropped_signals)):
        # calculate reconstructed signal
        rec_sig = reconstruct(magn_vectors, vs[i])
        rec_sigs.append(rec_sig)

        # calculate difference between original and reconstructed signals
        diffs, diff_x = calc_diff(cropped_signals[i], rec_sig, new_x, mag_is)

        # calculate averages
        ave = np.mean(diffs)
        aves.append(ave)
        all_diffs.append(diffs)

    ave_of_aves = np.mean(aves)

    return ave_of_aves, aves, all_diffs, rec_sigs, mag_is, cropped_signals, new_x


# from a set of signals, systematically  remove the most unphysical ones
# until a certain accuracy is reached. this is done by reconstructing
# optimal magnetic field vectors using pseudoinverse and calculating the total
# average difference between original signals and the reconstructed signals.
# when the total average goes below ave_sens, the calculation is stopped.
# reconstucted magnetic fields can be averaged for faster performance, but
# ave_sens MUST ALSO BE CHANGED for accurate calculation. returns the names
# of removed signals, the new x-values of the cropped signals (in indices)
# as well as the absolute and relative change in the average total diference
# at each removal
# the following ave_sens ave_window pairs are confirmed to produce good results:
# 10 ** (-13) 1
# 10 ** (-12) 100
def filter_unphysical_sigs(signals, names, xs, vs, bad_segs, ave_sens=10 ** (-13), ave_window=1):
    if len(signals) <= 3:
        print("too few signals, stopping")
        return [], [], [], []

    # crop signals
    cropped_signals, new_x = crop_all_sigs(signals, xs, bad_segs)

    print("analysing " + str(len(cropped_signals)) + " signals")
    temp_sigs = cropped_signals[:]
    temp_names = names[:]
    temp_vs = vs[:]

    # calculate initial reconstruction
    ave_of_aves, aves, diffs, rec_sigs, magis, cropped_signals, new_new_x = rec_and_diff(cropped_signals, [new_x], vs,
                                                                                         ave_window=ave_window)
    print("average at start:", ave_of_aves)

    if ave_of_aves < ave_sens:
        return [], new_x, [], []

    excludes = []
    ave_diffs = []
    rel_diffs = []
    while ave_of_aves > ave_sens:

        print(len(temp_sigs), "signals left")

        if len(temp_sigs) <= 3:
            print("no optimal magnetic field found")
            return [], new_x, [], []

        new_aves = []
        # calculate a new reconstruction excluding each signal one at a time
        for i in range(len(temp_sigs)):
            excl_name = temp_names[i]

            if excl_name in excludes:
                new_aves.append(100000)
                print("skipping", i)
                continue

            sigs_without = temp_sigs[:i] + temp_sigs[i + 1:]
            vs_without = temp_vs[:i] + temp_vs[i + 1:]

            # calculate reconstruction and average difference
            new_ave_of_aves, temp_aves, temp_diffs, temp_rec_sigs, temp_magis, temp_crop_sigs, temp_new_x = rec_and_diff(
                sigs_without, [new_x], vs_without, ave_window=ave_window)
            new_aves.append(new_ave_of_aves)

        # choose the lowest average difference and permanently exclude this signal
        # from the rest of the calculation
        best_ave = np.amin(new_aves)
        diff = ave_of_aves - best_ave
        rel_diff = diff / ave_of_aves
        print("average", best_ave)
        # print(ave_of_aves, best_ave, diff)
        ave_diffs.append(diff)
        rel_diffs.append(rel_diff)
        best_exclusion_i = new_aves.index(best_ave)
        ex_nam = temp_names[best_exclusion_i]
        print(ex_nam + " excluded")
        excludes.append(ex_nam)
        temp_vs.pop(best_exclusion_i)
        temp_sigs.pop(best_exclusion_i)
        temp_names.pop(best_exclusion_i)
        ave_of_aves = best_ave

    print(len(temp_names), "signals left at the end of calculation")
    return excludes, new_x, ave_diffs, rel_diffs


# goes through all detectors, does the filter_unphysical_sigs calculation for a
# signal cluster containing a given signal and all neighbouring signals and
# logs how many times each signal has been excluded and how many times it
# has been used in a filter_unphysical_sigs calculation. segments of the signals
# previously determined to be bad must also be included; the bad segments
# will be cropped out of the signals (if a signal cluster contains a bad
# segment, this segment will be removed from ALL SIGNALS IN THE CLUSTER).
# if the bad segments take up more than badness_sens of the signal, the
# signal will not be included in any calculation.
# the calculation is done on the gradients of the smoothed and filtered signals.
# returns a dictionary containing the names of the detectors as well as
# a dictionary of all the calculations it was included in and whether
# it was excluded. also returns the absolute and relative improvement
# each signal's exclusion caused to the average total difference.
def check_all_phys(signals, detecs, names, n_chan, bad_seg_list, smooth_window=401,
                   badness_sens=.5, ave_window=1, ave_sens=10 ** (-13)):
    import file_reader as fr

    def seg_lens(sig, segs):
        length = 0
        for seg in segs:
            length += seg[1] - seg[0]

        return length / len(sig)

    offset = int(smooth_window / 2)

    # initalise dictionaries
    all_diffs = {}
    all_rel_diffs = {}
    chan_dict = {}

    for name in names:
        all_diffs[name] = []
        all_rel_diffs[name] = []
        chan_dict[name] = {}

    for k in range(n_chan):
        # choose central detector and find nearby detectors
        comp_detec = names[k]
        print(comp_detec)

        nearby_names = find_nearby_detectors(comp_detec, detecs)
        nearby_names.append(comp_detec)

        # exclude signals whose bad segments are too long
        new_near = []
        for nam in nearby_names:
            index = names.index(nam)
            bad = seg_lens(signals[index], bad_seg_list[index]) > badness_sens

            if bad:
                print("excluding " + nam + " from calculation")
                continue

            new_near.append(nam)

        near_vs = []
        near_rs = []

        for name in new_near:
            near_vs.append(detecs[name][:3, 2])
            near_rs.append(detecs[name][:3, 3])

        near_sigs = fr.find_signals(new_near, signals, names)
        cluster_bad_segs = fr.find_signals(new_near, bad_seg_list, names)

        smooth_sigs = []
        xs = []

        # filter each signal, smooth and calculate gradient of smoothed and
        # filtered signals
        for i in range(len(near_sigs)):
            signal = near_sigs[i]
            filtered_signal, x, smooth_signal, smooth_x, new_smooth = hf.filter_and_smooth(signal, offset,
                                                                                           smooth_window)
            smooth_sigs.append(np.gradient(new_smooth))
            xs.append(x)

        # calculate which signals in the cluster to exclude
        exclude_chans, new_x, diffs, rel_diffs = filter_unphysical_sigs(smooth_sigs, new_near, xs, near_vs,
                                                                        cluster_bad_segs, ave_window=ave_window,
                                                                        ave_sens=ave_sens)

        if len(new_x) != 0:
            for nam in new_near:
                chan_dict[nam][comp_detec] = 0

        if len(new_x) > 1:
            print("analysed segment between", new_x[0], new_x[len(new_x) - 1])

        print("excluded", exclude_chans)

        # log and print data
        for j in range(len(exclude_chans)):
            chan = exclude_chans[j]
            diff = diffs[j]
            rel_diff = rel_diffs[j]
            all_diffs[chan].append(diff)
            all_rel_diffs[chan].append(rel_diff)
            chan_dict[chan][comp_detec] += 1

            tot = len(chan_dict[chan])
            ex = len([x for x in chan_dict[chan] if x == 1])

            print(chan, "times excluded:", ex, ", times in calculation:", tot,
                  ", fraction excluded:", float(ex / tot),
                  "average relative difference:", np.mean(all_rel_diffs[chan]))

        print()

    return all_diffs, all_rel_diffs, chan_dict


# analyse the data calculated by check_all_phys. if a signal has been excluded
# from a significant enough fraction (unphys_sensitivity),
# of all filter_unphysical_sigs calculations, it is marked as unphysical.
# if it has been included in too few calculations, it is marked as
# undetermined. if it has been included in no calculations,
# it is marked as unused. a confidence value is calculated for each marking;
# a higher value means a higher chance of the marking being correct.
# if a signal is marked as physical or unphysical but the confidence is below
# conf_sens, the signal is marked as undetermined.
# the confidence value is determined by the fraction of exclusions, number
# of times it has been included in a calculation and the differences caused
# by a signal's exclusion
# 0 = physical
# 1 = unphysical
# 2 = undetermined
# 3 = unused
def analyse_phys_dat(all_diffs, names, all_rel_diffs, chan_dict, frac_w=2.5,
                     diff_w=.5, num_w=.5, unphys_sensitivity=.25, conf_sens=.5,
                     min_chans=5):
    status = []
    confidence = []

    for i in range(len(names)):
        name = names[i]
        rel_diffs = all_rel_diffs[name]
        diffs = all_diffs[name]
        chan_dat = chan_dict[name]

        tot = np.float64(len(chan_dat))

        # mark signal that has been in a calculation too few times
        if tot == 0:
            status.append(3)
            confidence.append(3)
            continue

        if tot < min_chans:
            status.append(2)
            confidence.append(3)
            continue

        ex = np.float64(len([x for x in chan_dat if chan_dat[x] == 1]))
        frac_excluded = ex / tot

        ave_diff = np.mean(rel_diffs)

        # calculate confidence and mark
        if frac_excluded > unphys_sensitivity:
            stat = 1
            diff_conf = diff_w * ave_diff
            frac_conf = 1.25 * frac_w * frac_excluded
        else:
            stat = 0

            if math.isnan(ave_diff):
                diff_conf = diff_w
            else:
                diff_conf = diff_w * (1 - ave_diff)

            frac_conf = frac_w * (1 - frac_excluded / (2.5 * unphys_sensitivity))

        num_conf = num_w * tot / 14
        conf = (diff_conf + frac_conf + num_conf) / (diff_w + frac_w + num_w)

        if conf < conf_sens:
            stat = 2

        confidence.append(conf)
        status.append(stat)

    return status, confidence


# calculate absolute difference between points in two different signals.
# inputs may have different x-values with different spacings, as long as
# there is some overlap. x-values must be in indices
def calc_diff(signal1, signal2, x1, x2):
    x_min = max(np.amin(x1), np.amin(x2))
    x_max = min(np.amax(x1), np.amax(x2))
    new_x = list(range(x_min, x_max))
    new_new_x = []

    diffs = []

    for x in new_x:
        if x not in x1 or x not in x2:
            continue
        i1 = x1.index(x)
        i2 = x2.index(x)
        new_new_x.append(x)
        point1 = signal1[i1]
        point2 = signal2[i2]
        diffs.append(abs(point1 - point2))

    return diffs, new_new_x


# find detectors within a radius r_sens from a given detector.
def find_nearby_detectors(d_name, detectors, r_sens=0.06):
    dut = detectors[d_name]
    r_dut = dut[:3, 3]

    nears = []

    for name in detectors:
        if name == d_name:
            continue

        detector = detectors[name]
        r = detector[:3, 3]
        delta_r = np.sqrt((r_dut[0] - r[0]) ** 2 +
                          (r_dut[1] - r[1]) ** 2 +
                          (r_dut[2] - r[2]) ** 2)

        if delta_r < r_sens:
            nears.append(name)

    return nears


# filter the jump in the beginning of the signal. works better on good signals
def filter_start(signal, offset=50, max_rel=0.05):
    max_i = int(max_rel * len(signal))
    grad = np.gradient(signal[:max_i])
    max_grad_i = np.argmax(grad)
    min_grad_i = np.argmin(grad)
    farther_i = np.amax([max_grad_i, min_grad_i])
    return farther_i + offset


# check if the values of segments are close to eachother. previously calculated
# averages may also be included. difference in values is determined by
# calculating the standard deviation of all values.
def averages_are_close(signal, start_is, end_is, averages=None, std_sensitivity=0.015):
    if averages is None:
        averages = []

    if len(start_is) == 0:
        return False

    if len(start_is) == 1 and len(averages) == 0:
        return True

    for i in range(len(start_is)):
        segment = signal[start_is[i]: end_is[i]]
        av = np.mean(segment)
        averages.append(av)

    av_of_avs = sum(averages) / len(averages)
    std = np.std(averages) / abs(av_of_avs)
    return std <= std_sensitivity


# calculate the average value of the gradient of a signal between
# start_i and end_i. start_i may be offset if the start of a segment
# needs to be excluded
def average_of_gradient(signal, start_i, end_i, offset_percentage=0.05):
    length = end_i - start_i
    offset = int(offset_percentage * length)
    segment = signal[start_i + offset: end_i]
    grad = np.gradient(segment)
    return sum(grad) / len(grad)


# TODO fix confidences
# calculate a goodness value (a value determining how likely a flat segment
# has been wrongly detected by find_flat_segments) for a segment in a
# signal. a goodness value is increased if a segment has no clear trend,
# has a small fraction of unique values and is long.
# goodness > 1 -> bad
# goodness < 1 -> good/suspicious
# goodness < 0 -> very good
def cal_goodness_seg(signal, start_i, end_i,
                     uniq_w=1.5, grad_sensitivity=0.5 * 10 ** (-13),
                     grad_w=10 ** 12, len_w=1):
    segment = signal[start_i: end_i]
    uniqs = np.unique(segment)

    uniquevals = len(uniqs)
    totvals = len(segment)
    frac_of_uniq = 1 - uniquevals / totvals

    uniq_conf = uniq_w * frac_of_uniq

    grad_average = average_of_gradient(signal, start_i, end_i)

    if grad_average < grad_sensitivity:
        grad_conf = 0
    else:
        grad_conf = - grad_w * grad_average

    rel_len = (end_i - start_i) / len(signal)

    if rel_len >= .5:
        len_w = 1.5 * len_w

    len_conf = rel_len * len_w

    print("uniq_conf:", uniq_conf, "grad_conf:", grad_conf, "len_conf:", len_conf)

    tot_conf = uniq_conf + grad_conf + len_conf
    return tot_conf


# finds segments in the signal where the value stays approximately the same for long periods.
# returns lengths of segments, as well as their start and end indices.
# rel_sensitive_length determines how long a segment needs to be marked
# and relative_sensitivity determines how close the values need to be.
def find_flat_segments(signal, rel_sensitive_length=0.07, relative_sensitivity=0.02):
    lengths = []
    start_is = []
    end_is = []
    lock_val = None  # subsequent values are compared to this value

    sensitive_length = len(signal) * rel_sensitive_length
    length = 1
    for i in range(len(signal)):
        val = signal[i]

        if lock_val is None:
            is_close = False
        else:
            is_close = abs(abs(val - lock_val) / lock_val) < relative_sensitivity

        if not is_close or (is_close and i == len(signal) - 1):
            if length > sensitive_length:
                start_is.append(start_i)
                end_is.append(i)
                lengths.append(length)
            start_i = i
            length = 1
            lock_val = val

        if is_close:
            length += 1

    return lengths, start_is, end_is


# find segments where a certain value repeats. this filter ignores parts
# where the signal deviates from the unique value momentarily.
def uniq_filter_neo(signal, filter_i):
    uniqs, indices, counts = np.unique(signal[:], return_index=True, return_counts=True)
    max_repeat = np.amax(counts)
    if max_repeat <= 10:
        return [], []
    uniq_is = np.where(counts == max_repeat)

    max_vals = uniqs[uniq_is]
    where_repeat = np.where(signal == max_vals[0])
    where_repeat = list(where_repeat[0])
    where_repeat = [x for x in where_repeat if x > filter_i]

    if len(where_repeat) == 0:
        return [], []

    seg_start = np.amin(where_repeat)
    seg_end = np.amax(where_repeat)

    return [[seg_start, seg_end]], [2]


# reformat segment start and end indices into a single list
def reformat_segments(start_is, end_is):
    lst = []
    for i in range(len(start_is)):
        lst.append([start_is[i], end_is[i]])

    return lst


# find segments where the value stays the same value for a long period.
# also recalculates the tail of the signal and calculates
# a confidence value for the segment
def segment_filter_neo(signal):
    lengths, start_is, end_is = find_flat_segments(signal)

    if len(start_is) == 0:
        return [], []

    final_i = end_is[len(end_is) - 1]
    seg_is = reformat_segments(start_is, end_is)

    # recheck tail
    if final_i != len(signal) - 1:
        tail_ave = [np.mean(signal[final_i:])]
    else:
        tail_ave = []

    close = averages_are_close(signal, start_is, end_is, averages=tail_ave)

    if close:
        seg_is = [[start_is[0], end_is[len(end_is) - 1]]]

    confidences = []
    for segment in seg_is:
        confidences.append(cal_goodness_seg(signal, segment[0], segment[1]))

    return seg_is, confidences


# calculate a goodness value for a segment found by find_spikes. the confidence
# depends on the steepness of the spikes and their number,
# average gradient of the segment and the density of spikes.
# returns both the confidence and a segment that starts
# at the first spike and ends at the last.
# goodness > 1 -> bad
# goodness < 1 -> good
def cal_goodness_grad(gradient, spikes, all_diffs, max_sensitivities=None,
                      n_sensitivities=None,
                      grad_sensitivity=2 * 10 ** (-13),
                      sdens_sensitivity=0.1):
    if n_sensitivities is None:
        n_sensitivities = [20, 100]

    if max_sensitivities is None:
        max_sensitivities = [1.5, 1, .5]

    n = len(spikes)

    if n == 0:
        return [], None

    score = .5

    first_spike = spikes[0]
    seg_start = first_spike[0]
    last_spike = spikes[len(spikes) - 1]
    seg_end = last_spike[len(last_spike) - 1]
    seg_len = seg_end - seg_start

    if n == 1:
        return [seg_start, seg_end], .1

    spike_density = n / seg_len

    max_diffs = []
    for i in range(n):
        diffs = all_diffs[i]
        max_diffs.append(np.amax(diffs))

    av_max = np.mean(max_diffs)

    grad_ave = abs(np.mean(gradient[seg_start:seg_end]))

    # TEST DIFFS----------------------------------------
    if av_max >= max_sensitivities[0]:
        score += 2
    elif av_max >= max_sensitivities[1]:
        score += 1
    elif av_max >= max_sensitivities[2]:
        score += .5
    # --------------------------------------------------

    # TEST NUMBER OF SPIKES-----------------------------
    if n >= n_sensitivities[1]:
        score += 1
    elif n >= n_sensitivities[0]:
        score += .5
    # --------------------------------------------------

    # TEST GRADIENT-------------------------------------
    if grad_ave >= grad_sensitivity:
        score -= .25
    else:
        score += .5
    # --------------------------------------------------

    # TEST SPIKE DENSITY--------------------------------
    if spike_density >= sdens_sensitivity:
        score += 1

    score = score / 1.5

    print("num_spikes", n, "av_diff", av_max, "grad_ave", grad_ave,
          "spike_density", spike_density, "badness", score)

    return [seg_start, seg_end], score


# find spikes in the signal by checking where the absolute gradient of the signal
# abruptly goes above grad_sensitivity. returns the spikes and their
# difference between the gradient and the grad_sensitivity
def find_spikes(gradient, filter_i, grad_sensitivity, len_sensitivity=6):
    spikes = []
    all_diffs = []

    diffs = []
    spike = []
    for i in range(filter_i, len(gradient)):
        val = abs(gradient[i])

        if val > grad_sensitivity:
            spike.append(i)
            diffs.append((val - grad_sensitivity) / grad_sensitivity)
            continue

        if i - 1 in spike:
            if len(spike) < len_sensitivity:
                spikes.append(spike)
                all_diffs.append(diffs)

            spike = []
            diffs = []

    return spikes, all_diffs


# finds segments with steep spikes in the signal and calculates their goodness
def gradient_filter_neo(signal, filter_i, grad_sensitivity=10 ** (-10)):
    gradient = np.gradient(signal)
    spikes, all_diffs = find_spikes(gradient, filter_i, grad_sensitivity)
    seg_is, confidence = cal_goodness_grad(gradient, spikes, all_diffs)

    if len(seg_is) == 0:
        return [], []

    final_i = len(signal) - 1

    if seg_is[1] != final_i:
        tail = signal[seg_is[1]:]
        tail_ave = np.mean(tail)
        close = averages_are_close(signal, [seg_is[0]], [seg_is[1]], averages=[tail_ave])

        if close:
            seg_is[1] = final_i

    return [seg_is], [confidence]


# calculate the fft (absolute value) of the signal
def get_fft(signal, filter_i=0):
    from scipy.fft import fft

    if len(signal) == 0:
        return [0]

    ftrans = fft(signal[filter_i:])
    ftrans_abs = [abs(x) for x in ftrans]
    return ftrans_abs


# UNUSED
# calculate the fft for a windowed part of the signal. the window is scanned across
# the entire signal so that the fft values are as a function of time.
# returns the specified indices of the fft as a function of time.
# before calculating the ffts the trend of the signal is removed by first
# calculating a rolling average with a very large smoothing window and then
# getting the difference between the smoothed signal and the original.
def calc_fft_indices(signal, indices=None, window=400, smooth_window=401):
    if indices is None:
        indices = [1, 2, 6]

    sig_len = len(signal)
    ftrans_points = sig_len - window
    i_arr = np.zeros((len(indices), ftrans_points))

    # calculate smoothed signal
    offset = int(smooth_window / 2)
    smooth_signal = smooth(signal, window_len=smooth_window)
    smooth_x = [x - offset for x in list(range(len(smooth_signal)))]

    # calculate the x values for the smooth signal
    new_smooth = []
    for i in range(sig_len):
        new_smooth.append(smooth_signal[i + offset])

    filtered_signal = [a - b for a, b in zip(signal, new_smooth)]

    # calculate ffts
    for i in range(ftrans_points):
        end_i = i + window
        signal_windowed = filtered_signal[i: end_i]
        ftrans = get_fft(signal_windowed)

        for j in range(len(indices)):
            index = indices[j]
            i_arr[j][i] = ftrans[index]

    return i_arr, smooth_signal, smooth_x, filtered_signal


# UNUSED
# this function would be used to find the default value of the fft function
# calculated by the calc_fft_indices function
# scan a signal across the y axis and find the highest value segment which
# contains a significant amount of all the values in a signal.
def find_default_y(arr, num_points=5000, step=.1 * 10 ** (-7)):
    y_arr = np.linspace(0, 1 * 10 ** (-7), num_points)
    arr_len = len(arr)
    frac_arr = []

    # calculate the fraction of signals in a segment of lenght step as a
    # function of y
    for y_min in y_arr:
        y_max = y_min + step
        vals_in_step = [val for val in arr if y_min < val < y_max]
        frac = len(vals_in_step) / arr_len
        frac_arr.append(frac)

    # smooth the fraction function (this is done so that not every small
    # local maximum is detected)
    frac_arr = np.asarray(frac_arr)
    smooth_window = 201
    offset = int(smooth_window / 2)
    smooth_frac = smooth(frac_arr, window_len=smooth_window)
    smooth_x = [x - offset for x in list(range(len(smooth_frac)))]

    new_smooth = []
    for i in range(num_points):
        new_smooth.append(smooth_frac[i + offset])

    # find local extrema (whose value is above .1) of the smoothed fraction function
    from scipy.signal import argrelextrema
    new_smooth = np.asarray(new_smooth)
    smooth_max_is = argrelextrema(new_smooth, np.greater, order=10)[0]

    maxima = new_smooth[smooth_max_is]
    maxima = [val for val in maxima if val > .1]
    max_is = []

    for i in range(len(new_smooth)):
        val = new_smooth[i]
        if val in maxima:
            max_is.append(i)

    # return the highest of these maxima
    if len(max_is) == 0:
        final_i = None
        max_step = np.amax(frac_arr)
        max_i = list(frac_arr).index(max_step)
        seg_min = y_arr[max_i]
        seg_max = seg_min + step
    else:
        final_i = np.amax(max_is)
        seg_min = y_arr[final_i]
        seg_max = seg_min + step

    return y_arr, frac_arr, (seg_min, seg_max), new_smooth, max_is, final_i


# UNUSED
# this function would be used in conjunction with find_default_y to
# find the segments where the signal no longer is periodic.
def get_spans_from_fft(fft_i2, hseg, fft_window=400):
    segs = []
    all_fft_segs = []

    fft_segs = []
    temp_seg = [-1, -1]
    i_min = -1
    i_max = -1
    for fft_i in range(len(fft_i2)):
        fft_val = fft_i2[fft_i]
        val_in_hseg = hseg[0] < fft_val < hseg[1]

        if (i_max != -1 and i_min != - 1) and (not val_in_hseg or fft_i == len(fft_i2) - 1):
            print("in", i_min, i_max)
            if temp_seg[0] < i_min < temp_seg[1]:
                temp_seg = [temp_seg[0], i_max]
                print("between", temp_seg)
            elif temp_seg == [-1, -1]:
                temp_seg = [i_min, i_max]
                print("none", temp_seg)
            else:
                segs.append(temp_seg)
                print("append", temp_seg)
                temp_seg = [i_min, i_max]

            i_min = -1
            i_max = -1
            continue

        if val_in_hseg:
            i_min_temp = fft_i
            i_max_temp = fft_i + fft_window

            if i_min == -1:
                i_min = i_min_temp

            if i_max_temp > i_max:
                i_max = i_max_temp

    if temp_seg != [-1, -1]:
        print("final append")
        segs.append(temp_seg)

    return segs


# combine several segments so that there is no overlap between them
def combine_segments(segments):
    n = len(segments)

    if n == 0:
        return []

    segments_sorted = sorted(segments, key=itemgetter(0))

    combined_segs = []
    anchor_seg = segments_sorted[0]

    if n == 1:
        return segments

    for i in range(1, n):
        segment = segments_sorted[i]

        if anchor_seg[1] < segment[0]:
            combined_segs.append(anchor_seg)
            anchor_seg = segment

        new_start = anchor_seg[0]
        new_end = max(anchor_seg[1], segment[1])

        anchor_seg = [new_start, new_end]

        if i == n - 1:
            combined_segs.append(anchor_seg)

    return combined_segs


# sort segments into bad and suspicious based on their goodness values
def separate_segments(segments, confidences, conf_threshold=1):
    n = len(segments)

    bad_segs = []
    suspicious_segs = []

    for i in range(n):
        conf = confidences[i]
        segment = segments[i]

        if conf >= conf_threshold:
            bad_segs.append(segment)
        elif conf >= 0:
            suspicious_segs.append(segment)

    return bad_segs, suspicious_segs


# calculate length of all segments
def length_of_segments(segments):
    tot_len = 0
    for segment in segments:
        length = segment[1] - segment[0]
        tot_len += length

    return tot_len


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


# fix overlap between suspicious and bad segments. bad segments take priority
# over suspicious ones
def fix_overlap(bad_segs, suspicious_segs):
    if len(bad_segs) == 0 or len(suspicious_segs) == 0:
        return suspicious_segs

    new_suspicious_segs = []
    for sus_seg in suspicious_segs:
        sus_list = list(range(sus_seg[0], sus_seg[1] + 1))
        for bad_seg in bad_segs:
            bad_list = list(range(bad_seg[0], bad_seg[1] + 1))
            sus_list = list(set(sus_list) - set(bad_list))

        split_lists = split_into_lists(sus_list)
        split_segs = []

        for lst in split_lists:
            split_segs.append([np.amin(lst), np.amax(lst)])

        new_suspicious_segs += split_segs

    return new_suspicious_segs


# take all segments and their confidences and separate segments into good,
# suspicious and bad, as well as fix the overlap between them. the function
# also makes a decision whether or not the entire signal is concidered bad
# based on how much of the signal the bad segments take up
def final_analysis(signal_length, segments, confidences, badness_sensitivity=.8):
    bad_segs, suspicious_segs = separate_segments(segments, confidences)

    bad_segs = combine_segments(bad_segs)
    suspicious_segs = combine_segments(suspicious_segs)

    suspicious_segs = fix_overlap(bad_segs, suspicious_segs)

    tot_bad_length = length_of_segments(bad_segs)
    rel_bad_length = tot_bad_length / signal_length
    badness = rel_bad_length >= badness_sensitivity

    return badness, bad_segs, suspicious_segs


# go through all signals and determine suspicious and bad segments within them.
# this is done by running the signal through three different filters
# (gradient_filter_neo, segment_filter_neo and uniq_filter_neo).
# the function returns lists containing all bad and suspicious segments
# as well as ones containing whether the signal is bad (boolean value) and
# the time it took to analyse each signal.
def analyse_all_neo(signals, names, chan_num,
                    filters=None,
                    badness_sensitivity=.8):
    if filters is None:
        filters = ["uniq", "segment", "gradient"]

    exec_times = []
    signal_statuses = []
    bad_segment_list = []
    suspicious_segment_list = []

    for i in range(chan_num):
        print(names[i])
        signal = signals[i]
        signal_length = len(signal)
        segments = []
        confidences = []
        bad = False

        start_time = time.time()
        filter_i = filter_start(signal)

        for fltr in filters:
            print("beginning analysis with " + fltr + " filter")

            if fltr == "uniq":
                seg_is, confs = uniq_filter_neo(signal, filter_i)

            if fltr == "segment":
                seg_is, confs = segment_filter_neo(signal)

            if fltr == "gradient":
                seg_is, confs = gradient_filter_neo(signal, filter_i)

            new_segs = len(seg_is)

            if new_segs == 0:
                print("no segments found")
            else:
                print(new_segs, "segment(s) found")

            segments += seg_is
            confidences += confs

        bad, bad_segs, suspicious_segs = final_analysis(signal_length, segments, confidences,
                                                        badness_sensitivity=badness_sensitivity)
        num_bad = len(bad_segs)
        num_sus = len(suspicious_segs)
        print(num_sus, "suspicious and", num_bad, " bad segment(s) found in total")

        if not bad:
            print("signal not marked as bad")

        signal_statuses.append(bad)
        bad_segment_list.append(bad_segs)
        suspicious_segment_list.append(suspicious_segs)

        end_time = time.time()
        exec_time = end_time - start_time
        print("execution time:", exec_time)
        exec_times.append(exec_time)

        print()

    return signal_statuses, bad_segment_list, suspicious_segment_list, exec_times
