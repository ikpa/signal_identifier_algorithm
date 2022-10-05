import argparse
import time
import signal_analysis as sa
import file_reader as fr
import helper_funcs as hf
import numpy as np

import test_funcs as tf


# get command line arguments
def arg_parser():
    parser = argparse.ArgumentParser(description="analyse a meg dataset and"
                                                 "find faulty and/or unphysical"
                                                 "measurements")

    parser.add_argument("--filename", required=True, type=str, help="filename of the dataset to analyse")
    parser.add_argument('--filters', nargs='+', choices=["uniq", "segment", "spike"],
                        default=["uniq", "segment", "spike"], help="the basic filters to use")
    parser.add_argument("-p", "--physicality", action="store_true", default=False, help="do physicality analysis")
    parser.add_argument("--plot", action="store_true", default=False, help="plot signals with results")
    return parser.parse_args()


# print results
def print_results(names, signals, filt_statuses, bad_segs, suspicious_segs,
                  phys_statuses=[], phys_confs=[],
                  all_rel_diffs=[], chan_dict=[]):
    print_phys = not len(phys_statuses) == 0
    print("results:")
    print()

    for i in range(len(names)):
        name = names[i]
        signal = signals[i]
        filt_status = filt_statuses[i]
        bad_seg = bad_segs[i]
        sus_seg = suspicious_segs[i]
        print(name)

        if filt_status:
            print("signal marked as bad")
        else:
            print("signal not marked as bad")

        sig_len = len(signal)
        bad_len = sa.length_of_segments(bad_seg)
        sus_len = sa.length_of_segments(sus_seg)

        rel_bad_len = bad_len / sig_len
        rel_sus_len = sus_len / sig_len
        print("fraction of signal marked as bad: " + str(rel_bad_len) + ", bad segments:", bad_seg)
        print("fraction of signal marked as suspicious: " + str(rel_sus_len) + ", suspicious segments:", sus_seg)

        if print_phys:
            phys_stat = phys_statuses[i]
            phys_conf = phys_confs[i]
            rel_diffs = all_rel_diffs[name]
            chan_dat = chan_dict[name]

            times_in_calc = len(chan_dat)
            times_ex = len([x for x in chan_dat if chan_dat[x] == 1])

            ave_diff = np.mean(rel_diffs)

            if phys_stat == 0:
                phys_string = "signal determined to be physical"
            if phys_stat == 1:
                phys_string = "signal determined to be unphysical"
            if phys_stat == 2:
                phys_string = "physicality of signal undetermined"
            if phys_stat == 3:
                phys_string = "signal not used in physicality calculation"

            print(phys_string + ", confidence: " + str(phys_conf))
            print("times in calculation: " + str(times_in_calc) + ", times excluded: " +
                  str(times_ex) + ", average relative improvement when excluded: " + str(ave_diff))

        print()


# current version of the program.
def thirdver(fname, filters, phys, plot):
    detecs = np.load("array120_trans_newnames.npz")

    print("analysing " + fname)
    print()

    signals, names, t, n_chan = fr.get_signals(fname)
    print("beginning analysis with the following filters:", filters)
    print()
    start_time = time.time()
    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(signals, names, n_chan, filters=filters)
    end_time = time.time()
    filt_time = (end_time - start_time) / 60
    print("time elapsed in filtering: " + str(filt_time) + " mins")
    print()

    if phys:
        print("-----------------------------------------------------")
        print("beginning physicality analysis")
        print()
        start_time = time.time()
        # ave_sens = 10**(-12)
        all_diffs, all_rel_diffs, chan_dict = sa.check_all_phys(signals, detecs, names, n_chan, bad_segs,
                                                                ave_window=100, ave_sens=5 * 10 ** (-13))

        phys_stat, phys_conf = sa.analyse_phys_dat(all_diffs, names, all_rel_diffs, chan_dict)
        end_time = time.time()
        phys_time = (end_time - start_time) / 60
        print()
        print("time elapsed in physicality analysis: " + str(phys_time) + " mins")
    else:
        phys_stat = []
        phys_conf = []
        all_rel_diffs = []
        chan_dict = []
        phys_time = 0

    tot_time = phys_time + filt_time
    print("-----------------------------------------------------")
    print()
    print_results(names, signals, signal_statuses, bad_segs, suspicious_segs,
                  phys_stat, phys_conf, all_rel_diffs, chan_dict)
    print("total time elapsed: " + str(tot_time) + " mins")

    if plot:
        hf.plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, physicality=phys_stat)


def main():
    args = arg_parser()
    thirdver(args.filename, args.filters, args.physicality, args.plot)


if __name__ == '__main__':
    main()
    #tf.test_new_excluder()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
