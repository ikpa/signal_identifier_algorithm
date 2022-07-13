import ruptures as rpt
import numpy as np
import pandas as pd
import time

bkps = 8

# fixes signals in bad formats (when reading only one channel)
def reorganize_signal(signal):
    size = len(signal)
    new_sig = np.zeros(size)
    # print(size)

    for i in range(size):
        new_sig[i] = signal[i][0]

    return new_sig

def reorganize_signals(signals, n):
    new_signals = []
    for i in range(n):
        signal = signals[:, i]
        new_signals.append(signal)

    return new_signals

def find_nearby_detectors(d_name, detectors, r_sens = 0.06):
    dut = detectors[d_name]
    r_dut = dut[:3, 3]
    v_dut = dut[:3, 2]

    nears = []

    for name in detectors:
        if name == d_name or name.endswith("4"):
            continue

        detector = detectors[name]
        r = detector[:3, 3]
        delta_r = np.sqrt((r_dut[0] - r[0])**2 +
                          (r_dut[1] - r[1])**2 +
                          (r_dut[2] - r[2])**2)

        if delta_r < r_sens:
            nears.append(name)

    return nears


# filter the jump in the beginning of the signal. works better on good signals
#TODO fix
def filter_start(signal, offset=50, max_rel=0.05):
    max_i = int(max_rel * len(signal))
    grad = np.gradient(signal[:max_i])
    max_grad_i = np.argmax(grad)
    min_grad_i = np.argmin(grad)
    farther_i = np.amax([max_grad_i, min_grad_i])
    return farther_i + offset


# check if the values of segments are close to eachother.
def averages_are_close(signal, start_is, end_is, averages=[], std_sensitivity=0.015):
    if len(start_is) == 0:
        return False

    #print(end_is)
    #print(averages)
    if len(start_is) == 1 and len(averages) == 0:
        return True

    print("checking averages")

    # print("start is", len(start_is))
    # print(start_is)
    for i in range(len(start_is)):
        # print(i)
        segment = signal[start_is[i]: end_is[i]]
        av = sum(segment) / len(segment)
        # print(averages)
        averages.append(av)

    # print(averages)
    av_of_avs = sum(averages) / len(averages)
    std = np.std(averages) / abs(av_of_avs)
    print("std", std)
    return std <= std_sensitivity


# UNUSED
def find_start_of_seg(signal, end_i):
    averages = np.zeros_like(signal)
    averages[end_i] = signal[end_i]

    for i in range(end_i - 1, -1, -1):
        seg = signal[i: end_i]
        ave = sum(seg) / len(seg)
        averages[i] = ave

    return averages


def average_of_gradient(signal, start_i, end_i, offset_percentage=0.05):
    length = end_i - start_i
    offset = int(offset_percentage * length)
    segment = signal[start_i + offset: end_i]
    grad = np.gradient(segment)
    return sum(grad) / len(grad)

def cal_confidence_seg(signal, start_i, end_i,
                       uniq_w=2, grad_sensitivity=0.5 * 10 ** (-13),
                       grad_w=10**12):
    segment = signal[start_i : end_i]
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

    print("uniq_conf:", uniq_conf)
    print("grad_conf:", grad_conf)

    tot_conf = uniq_conf + grad_conf
    return tot_conf

#TODO check values and CHECK LENGTH ELSEWHERE
def cal_confidence(signal, start_i, end_i,
                   len_w=1.5, grad_w=10 ** 12, dist_w=1,
                   grad_sensitivity=0.5 * 10 ** (-13)):
    grad_average = average_of_gradient(signal, start_i, end_i)
    rel_len = (end_i - start_i) / len(signal)
    rel_dist = ((len(signal) - 1) - end_i) / len(signal)

    if grad_average < grad_sensitivity:
        grad_conf = 0
    else:
        grad_conf = - grad_w * grad_average

    len_conf = len_w * rel_len
    dist_conf = - dist_w * rel_dist

    print("grad conf", grad_conf)
    print("len conf", len_conf)
    print("dist conf", dist_conf)

    confidence = grad_conf + len_conf + dist_conf

    return confidence

# the proper one for now
# further analyse segments which stay around one value for too long
def multi_seg_analysis(signal, start_is, end_is, badness_sensitivity):
    final_i = len(signal) - 1
    final_segment_i = end_is[len(end_is) - 1]

    end_of_segment = final_segment_i

    if final_i != final_segment_i:
        segment = signal[final_segment_i:]
        ave = sum(segment) / len(segment)

        if averages_are_close(signal, start_is, end_is, averages=[ave]):
            end_of_segment = final_i

    tot_length = end_of_segment - start_is[0]
    same_frac = tot_length / len(signal)
    bad = same_frac > badness_sensitivity
    print("same_frac", same_frac)
    # print("bad", bad)

    if bad:
        return (bad, 2), end_of_segment

    confidence = cal_confidence(signal, start_is[0], end_of_segment)

    return (bad, confidence), end_of_segment

#220
# finds segments in the signal where the value stays approximately the same for long periods
def find_uniq_segments(signal, rel_sensitive_length = 0.07, relative_sensitivity=0.02):
    lengths = []
    start_is = []
    end_is = []
    lock_val = None

    sensitive_length = len(signal) * rel_sensitive_length
    length = 1
    for i in range(len(signal)):
        val = signal[i]

        if lock_val == None:
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

#TODO fix things with filter_i
def uniq_filter_neo(signal, filter_i):
    uniqs, indices, counts = np.unique(signal[filter_i:], return_index=True, return_counts=True)
    print(counts)
    print(np.amax(counts))
    max_repeat = np.amax(counts)
    if max_repeat <= 10:
        return []
    uniq_is = np.where(counts == max_repeat)

    max_vals = uniqs[uniq_is]
    print(max_vals)
    where_repeat = np.where(signal == max_vals[0])

    #where_repeat = [x + filter_i for x in where_repeat]

    return where_repeat

# check the percentage of the signal where the value stays exactly the same
def uniq_filter(signal, sensitivity=0.2):
    uniqs, indices, counts = np.unique(signal, return_index=True, return_counts=True)
    uniq_is = np.where(counts > 1)
    max_vals = uniqs[uniq_is]

    where_repeat = []
    for max_val in max_vals:
        i_arr = np.where(signal == max_val)
        where_repeat.append(i_arr)

    uniquevals = len(uniqs)
    totvals = len(signal)
    frac_of_uniq = uniquevals / totvals

    #return frac_of_uniq, indices, counts[uniq_is], where_repeat, (frac_of_uniq <= sensitivity, 2)
    bad = frac_of_uniq <= sensitivity
    if bad:
        confidence = 2
    else:
        confidence = 0
    return frac_of_uniq, where_repeat, (bad, confidence)

def reformat_stats(start_is, end_is):
    list = []
    for i in range(len(start_is)):
        list.append([start_is[i], end_is[i]])

    return list

def segment_filter_neo(signal, badness_sensitivity=0.8):
    lengths, start_is, end_is = find_uniq_segments(signal)

    if len(start_is) == 0:
        return False, [], [None]

    final_i = end_is[len(end_is) - 1]
    seg_is = reformat_stats(start_is, end_is)

    #recheck tail
    if final_i != len(signal) - 1:
        tail_ave = [np.mean(signal[final_i:])]
    else:
        tail_ave = []

    close = averages_are_close(signal, start_is, end_is, averages=tail_ave)
    if close:
        seg_is = [[start_is[0], end_is[len(end_is) - 1]]]
        length = end_is[len(end_is) - 1] - start_is[0]
        print(length / len(signal))

        if length / len(signal) >= badness_sensitivity:
            return True, seg_is, [None]

    confidences = []
    #print(seg_is)
    for segment in seg_is:
        #print(segment)
        confidences.append(cal_confidence_seg(signal, segment[0], segment[1]))

    return False, seg_is, confidences

#confidence 2 = bad
#confidence < 0.01 = good
#confidence 1.5 - 0.01 = sus
#TODO figure out how to use where_repeat
def segment_filter(signal, where_repeat, badness_sensitivity=0.8,
                   confidence_sensitivity=0.01):
    lengths, start_is, end_is = find_uniq_segments(signal)

    same_sum = sum(lengths)
    same_frac = same_sum / len(signal)

    prebad = same_frac > badness_sensitivity

    stats = []

    if not prebad:
        if len(start_is) == 0:
            print("no segments, good")
            bad = (prebad, 0)

        if len(start_is) == 1:
             print("one uniq segment, rechecking tail with multiseg")
        #     bad, new_end = multi_seg_analysis(signal, start_is, end_is, badness_sensitivity)
        #     end_is = new_end
        #     #start_is = start_is[0]
        #     stats = [[start_is[0], new_end]]

        if len(start_is) >= 1:
            print(str(len(start_is)) + " uniq segments, checking similarity")

            if averages_are_close(signal, start_is, end_is, averages=[]):
                print("averages close, doing multiseg")
                bad, new_end = multi_seg_analysis(signal, start_is, end_is, badness_sensitivity)
                #start_is = start_is[0]
                #end_is = new_end
                stats = [[start_is[0], new_end]]
            else:
                print("averages not close, no multiseg")
                bad = (prebad, 0)
                stats = reformat_stats(start_is, end_is)

    else:
        print("length of segments over sensitivity level, bad")
        bad = (prebad, 2)
        stats = []

    return stats, bad

def cal_confidence_grad(gradient, spikes, all_diffs, max_sensitivities=[1.5, 1, .5],
                   n_sensitivities=[20, 100],
                   grad_sensitivity=2 * 10 ** (-13),
                        sdens_sensitivity=0.1):
    n = len(spikes)

    if n == 0:
        return [], 0

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

    print("num_spikes", n, "av_diff", av_max, "grad_ave", grad_ave,
          "spike_density", spike_density)

    # TEST DIFFS----------------------------------------
    if av_max >= max_sensitivities[0]:
        score += 2
    elif av_max >= max_sensitivities[1]:
        score += 1
    elif av_max >= max_sensitivities[2]:
        score += .5
    # --------------------------------------------------

    #TEST NUMBER OF SPIKES-----------------------------
    if n >= n_sensitivities[1]:
        score += 1
    elif n >= n_sensitivities[0]:
        score += .5
    #--------------------------------------------------

    # TEST GRADIENT-------------------------------------
    if grad_ave >= grad_sensitivity:
        score -= .25
    else:
        score += .5
    # --------------------------------------------------

    # TEST SPIKE DENSITY--------------------------------
    if spike_density >= sdens_sensitivity:
        score += 1

    print("score", score)

    return [seg_start, seg_end], score / 1.5


#TODO score should represent segment only, relative length of bad segment
#TODO determines badness of signal
def analyse_spikes(gradient, spikes, all_diffs, max_sensitivities=[1.5, 1, .5],
                   n_sensitivities=[20, 100], seg_sensitivity=.2,
                   grad_sensitivity=2 * 10 ** (-13)):
    n = len(spikes)
    length = len(gradient)

    if n == 0:
        return [], (False, 0)

    score = .5

    first_spike = spikes[0]
    seg_start = first_spike[0]
    last_spike = spikes[len(spikes) - 1]
    seg_end = last_spike[len(last_spike) - 1]
    seg_len = (seg_end - seg_start) / length

    max_diffs = []
    for i in range(n):
        diffs = all_diffs[i]
        max_diffs.append(np.amax(diffs))

    av_max = np.mean(max_diffs)

    print("num_spikes", n, "av_diff", av_max, "seg_len", seg_len)

    #TEST DIFFS----------------------------------------
    if av_max >= max_sensitivities[0]:
        score += 3
        #bad
        return [seg_start, seg_end], (True, score)

    if av_max >= max_sensitivities[1]:
        score += 1

    if av_max >= max_sensitivities[2] and av_max <= max_sensitivities[1]:
        score += .5
    #--------------------------------------------------

    #TEST NUMBER OF SPIKES-----------------------------
    if n <= n_sensitivities[0]:
        #good
        return [seg_start, seg_end], (False, score)

    if n >= n_sensitivities[1]:
        score += 1
    else:
        score += .5
    #--------------------------------------------------

    #TEST LENGTH OF SEGMENT ---------------------------
    if seg_len <= seg_sensitivity:
        #bad
        return [seg_start, seg_end], (True, score)

    score += .5
    #--------------------------------------------------

    grad_ave = abs(np.mean(gradient[seg_start:seg_end]))
    print("grad_ave", grad_ave)

    #TEST GRADIENT-------------------------------------
    if grad_ave >= grad_sensitivity:
        #good
        score -= .5
        return [seg_start, seg_end], (False, score)

    score += .5
    #--------------------------------------------------

    return [seg_start, seg_end], (True, score)

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
                all_diffs.append(diffs )

            spike = []
            diffs = []

    return spikes, all_diffs

def gradient_filter_neo(signal, filter_i, grad_sensitivity=10 ** (-10)):
    gradient = np.gradient(signal)
    spikes, all_diffs = find_spikes(gradient, filter_i, grad_sensitivity)
    seg_is, confidence = cal_confidence_grad(gradient, spikes, all_diffs)

    if len(seg_is) == 0:
        return [], (False, confidence)

    return [seg_is], (False, confidence)

def gradient_filter(signal, filter_i, grad_sensitivity=10 ** (-10)):
    gradient = np.gradient(signal)
    spikes, all_diffs = find_spikes(gradient, filter_i, grad_sensitivity)
    seg_is, bad = analyse_spikes(gradient, spikes, all_diffs)
    bad = list(bad)
    bad[1] = bad[1] / 1.75

    if len(seg_is) == 0:
        return [], bad

    return [seg_is], bad

def append_stats(list, stats):
    if len(stats) == 0:
        return False

    for item in stats:
        list.append(item)
    return True

def combine_bads(bad_list, confidence_sensitivity=10):
    confidence = 0
    final_bad = False
    for bad in bad_list:
        print(bad[1])
        confidence += bad[1]

        if bad[0]:
            final_bad = True

    if confidence >= confidence_sensitivity:
        final_bad = True

    return (final_bad, confidence)

#TODO finish (check all segments if signal passes all filters)
def analyse_all_neo(signals, names, chan_num,
                    filters=["uniq", "segment", "gradient"]):
    exec_times = []
    signal_statuses = []
    segment_list = []
    confidence_list = []

    for i in range(chan_num):
        print(names[i])
        signal = signals[i]
        segments = []
        confidences = []
        bad = False

        start_time = time.time()
        filter_i = filter_start(signal)

        for filter in filters:
            print("beginning analysis with " + filter + " filter")

            if filter == "uniq":
                frac_of_uniq, where_repeat, result = uniq_filter(signal)
                bad = result[0]
                confidences.append(None)
                segments.append([])

            if filter == "segment":
                bad, seg_is, seg_confs = segment_filter_neo(signal)
                append_stats(segments, seg_is)
                append_stats(confidences, seg_confs)

            if filter == "gradient":
                seg_is, result = gradient_filter_neo(signal, filter_i)
                append_stats(segments, seg_is)
                confidence = result[1]
                bad = result[0]
                #print(confidence)
                confidences.append(confidence)

            if bad:
                print("bad signal, stopping analysis")
                break

        print(segments)
        print(confidences)
        segment_list.append(segments)
        signal_statuses.append(bad)
        confidence_list.append(confidences)

        end_time = time.time()
        exec_time = end_time - start_time
        print("execution time:", exec_time)
        exec_times.append(exec_time)

        print()

    return signal_statuses, segment_list, confidence_list, exec_times

#TODO FIX
def analyse_all(signals, names, chan_num):
    exec_times = []
    signal_status = []
    fracs = []
    uniq_stats_list = []
    for i in range(chan_num):
        print(names[i])
        signal = signals[i]

        segment_stats = []

        start_time = time.time()
        filter_i = filter_start(signal)

        bad_list = []

        print("testing unique values")
        frac_of_uniq, where_repeat, bad = uniq_filter(signal)
        bad_list.append(bad)

        if not bad[0]:
            print("checking for repetitive segments")
            stats, bad = segment_filter(signal, where_repeat)
            append_stats(segment_stats, stats)
            bad_list.append(bad)
        else:
            print("not enough unique values, bad")

        if not bad[0]:
            print("checking for spikes in gradient")
            stats, bad = gradient_filter(signal, filter_i)
            append_stats(segment_stats, stats)
            bad_list.append(bad)
            # if not len(stats) == 0:
            #     segment_stats.append(stats)

        else:
            print("too many repetitive segments, bad")

        #print(segment_stats)

        bad = combine_bads(bad_list)

        end_time = time.time()
        exec_time = end_time - start_time
        print("execution time:", exec_time)

        exec_times.append(exec_time)
        signal_status.append(bad)
        fracs.append(frac_of_uniq)
        uniq_stats_list.append(segment_stats)

        print()

    return signal_status, fracs, uniq_stats_list, exec_times


def pd_get_changes(signal):
    print(signal)


def find_changes_rpt(signal, method):
    if method == "Pelt":
        algo = rpt.Pelt().fit(signal)
        points = algo.predict(pen=0.01)

    if method == "Dynp":
        algo = rpt.Dynp().fit(signal)
        points = algo.predict(n_bkps=bkps)

    if method == "Binseg":
        algo = rpt.Binseg().fit(signal)
        points = algo.predict(n_bkps=bkps)

    if method == "Window":
        algo = rpt.Window().fit(signal)
        points = algo.predict(n_bkps=bkps)

    return points


def rpt_plot(signal, points):
    rpt.display(signal, points)
