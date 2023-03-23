import argparse
import time

import matplotlib.pyplot as plt

import pca
import fdfs as sa
import file_handler as fr
import helper_funcs as hf
import numpy as np

import warnings

import test_funcs as tf

default_filters = ["uniq", "flat", "spike", "fft"]
print_modes = ["print", "file", "none"]


# get command line arguments
def arg_parser():
    parser = argparse.ArgumentParser(description="analyse a meg dataset and"
                                                 "find faulty and/or unphysical"
                                                 "measurements")

    parser.add_argument("--filename", required=True, type=str, help="filename of the dataset to analyse")
    parser.add_argument('--filters', nargs='+', choices=default_filters,
                        default=default_filters, help="the basic filters to use")
    parser.add_argument("-p", "--physicality", action="store_true", default=False, help="do physical consistency analysis")
    parser.add_argument("--plot", action="store_true", default=False, help="plot signals with results")
    parser.add_argument("-prnt", "--print_mode", default="print", choices=print_modes)
    parser.add_argument("-log", "--log_filename", default="")
    return parser.parse_args()


# print results
def print_results(names, signals, filt_statuses, bad_segs, suspicious_segs, printer,
                  phys_statuses=[], phys_confs=[],
                  all_rel_diffs=[], chan_dict=[]):
    print_phys = not len(phys_statuses) == 0
    printer.extended_write("results:")
    printer.extended_write()

    for i in range(len(names)):
        # print(len(names))
        name = names[i]
        signal = signals[i]
        filt_status = filt_statuses[i]
        bad_seg = bad_segs[i]
        sus_seg = suspicious_segs[i]
        printer.extended_write(name)

        if filt_status:
            printer.extended_write("signal marked as bad")
        else:
            printer.extended_write("signal not marked as bad")

        sig_len = len(signal)
        bad_len = hf.length_of_segments(bad_seg)
        sus_len = hf.length_of_segments(sus_seg)

        rel_bad_len = bad_len / sig_len
        rel_sus_len = sus_len / sig_len
        printer.extended_write("fraction of signal marked as bad: " + str(rel_bad_len) + ", bad segments:", bad_seg)
        printer.extended_write("fraction of signal marked as suspicious: " + str(rel_sus_len) + ", suspicious segments:", sus_seg)

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

            printer.extended_write(phys_string + ", confidence: " + str(phys_conf))
            printer.extended_write("times in calculation: " + str(times_in_calc) + ", times excluded: " +
                  str(times_ex) + ", average relative improvement when excluded: " + str(ave_diff))

        printer.extended_write()


# current version of the program.
def thirdver(fname, filters, phys, print_mode, log_fname):
    detecs = np.load("array120_trans_newnames.npz")

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if print_mode == "file":
        file = open(log_fname, "w")
        printer = fr.Printer("file", file)
    else:
        printer = fr.Printer(print_mode)

    printer.extended_write("analysing " + fname, additional_mode="print")
    printer.extended_write("", additional_mode="print")


    signals, names, t, n_chan = fr.get_signals(fname)
    printer.extended_write("beginning filtering with the following filters:", filters, additional_mode="print")
    printer.extended_write("", additional_mode="print")
    start_time = time.time()
    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(signals, names, n_chan, printer, filters=filters, fft_goertzel=False)
    end_time = time.time()
    filt_time = (end_time - start_time)
    printer.extended_write("time elapsed in filtering: " + str(filt_time) + " secs", additional_mode="print")
    printer.extended_write()

    if phys:
        printer.extended_write("-----------------------------------------------------", additional_mode="print")
        printer.extended_write("beginning physicality analysis", additional_mode="print")
        printer.extended_write("", additional_mode="print")
        start_time = time.time()
        # ave_sens = 10**(-12)
        all_diffs, all_rel_diffs, chan_dict = pca.check_all_phys(signals, detecs, names, n_chan, bad_segs, suspicious_segs, printer,
                                                                 ave_window=100, ave_sens=5 * 10 ** (-13))

        phys_stat, phys_conf = pca.analyse_phys_dat_alt(all_diffs, names, all_rel_diffs, chan_dict)
        end_time = time.time()
        phys_time = (end_time - start_time)
        printer.extended_write()
        printer.extended_write("time elapsed in physicality analysis: " + str(phys_time) + " secs", additional_mode="print")
    else:
        phys_stat = []
        phys_conf = []
        all_rel_diffs = []
        chan_dict = []
        phys_time = 0

    tot_time = phys_time + filt_time
    printer.extended_write("-----------------------------------------------------", additional_mode="print")
    printer.extended_write("", additional_mode="print")

    #print_results(names, signals, signal_statuses, bad_segs, suspicious_segs, printer,
    #                    phys_stat, phys_conf, all_rel_diffs, chan_dict)

    i_x = list(range(len(t)))
    bad_segs_time = hf.segs_from_i_to_time(i_x, t, bad_segs)
    sus_segs_time = hf.segs_from_i_to_time(i_x, t, suspicious_segs)
    col_names = ["name", "bad segments", "suspicious segments"]
    write_data = [names, bad_segs_time, sus_segs_time]

    if phys:
        write_data.append(phys_stat)
        write_data.append(phys_conf)
        col_names.append("pca status")
        col_names.append("pca fraction")


    printer.extended_write("total time elapsed: " + str(tot_time) + " secs", additional_mode="print")

    if print_mode == "file":
        file.close()

    plot_dat = [signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, phys_stat, t]

    return col_names, write_data, plot_dat


# TODO test with larger values of seg_extend
def partial_analysis(time_seg, fname, print_mode, output="output_test.txt", log_fname="test.log", channels=["MEG*1", "MEG*4"],
                     filters=default_filters, seg_extend=200, phys=False):
    signals, names, t, n_chan = fr.get_signals(fname, channels=channels)

    if print_mode == "file":
        printer = fr.Printer("file", open(log_fname, "w"))
    else:
        printer = fr.Printer(print_mode)

    cropped_signals, cropped_ix, seg_i = hf.crop_signals_time(time_seg, t, signals, seg_extend)
    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(cropped_signals, names, n_chan, printer, filters=filters, filter_beginning=False)
    detecs = np.load("array120_trans_newnames.npz")
    good_seg_list = hf.find_good_segs(seg_i, bad_segs, cropped_ix[0])

    if phys:
        all_diffs, all_rel_diffs, chan_dict = pca.check_all_phys(cropped_signals, detecs, names, n_chan, bad_segs,
                                                                 suspicious_segs, printer,
                                                                 ave_window=100, ave_sens=5 * 10 ** (-13))

        phys_stat, phys_conf = pca.analyse_phys_dat_alt(all_diffs, names, all_rel_diffs, chan_dict)

    good_segs_time = hf.segs_from_i_to_time(cropped_ix, t, good_seg_list)

    col_names = ["name", "good segments"]
    write_data = [names, good_segs_time]

    if phys:
        write_data.append(phys_stat)
        write_data.append(phys_conf)
        col_names.append("pca status")
        col_names.append("pca fraction")

    if output is not None:
        fr.write_data_compact(output, write_data, col_names)

    return col_names, write_data

    # if False:
    #     for i in range(n_chan):
    #         i_x = cropped_ix[i]
    #         t_x = t[i_x]
    #         name = names[i]
    #         bad_seg_plot = bad_segs_time[i]
    #         sus_segs_plot = sus_segs_time[i]
    #         good_segs_plot = good_segs_time[i]
    #         cropped_sig = cropped_signals[i]
    #         p_stat = phys_stat[i]
    #         p_conf = phys_conf[i]
    #         figure, ax = plt.subplots()
    #         plt.plot(t_x, cropped_sig)
    #         hf.plot_spans(ax, bad_seg_plot, color="red")
    #         hf.plot_spans(ax, sus_segs_plot, color="yellow")
    #         hf.plot_spans(ax, good_segs_plot, color="green")
    #
    #         ax.axvline(t[seg_i[0]], linestyle="--", color="black")
    #         ax.axvline(t[seg_i[-1]], linestyle="--", color="black")
    #
    #         status = name + ", " + str(good_segs_plot)
    #
    #         if phys:
    #             status += ", " + str(p_stat) + ", " + str(p_conf)
    #
    #         ax.set_title(status)
    #         plt.show()


def main():
    args = arg_parser()

    if args.print_mode == "file":
        if args.log_filename == "":
            logfname = "log.log"
        else:
            logfname = args.log_filename

    else:
        logfname = ""

    col_names, data, plot_dat = thirdver(args.filename, args.filters, args.physicality, args.print_mode, logfname)

    signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, phys_stat, t = plot_dat

    fr.write_data_compact("output_test.txt", data, col_names)

    if args.plot:
        hf.plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, physicality=phys_stat, time_x=t)


if __name__ == '__main__':
    #main()
    # tf.test_fft()
    # tf.show_helmet()
    #tf.test_fft_emergency()
    # tf.show()
    # tf.test_new_excluder()
    # tf.test_magn2()
    # tf.test_seg_finder()
    #tf.test_crop()
    #tf.test_ffft()
    tf.show_pca()
    # tf.test_flat_new()
    #datadir = "example_data_for_patrik/"
    #partial_analysis([0.3, 0.36], datadir + "many_many_successful2.npz", "print", phys=True)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
