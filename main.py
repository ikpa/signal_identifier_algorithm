import argparse
import time

import signal_generator as sg
import signal_analysis as sa
import matplotlib.pyplot as plt
import file_reader as fr
import helper_funcs as hf
import test_funcs as tf
import numpy as np
import pandas as pd
import helmet_vis as vis
from mayavi import mlab
import re

datadir = "example_data_for_patrik/"


def arg_parser():
    parser = argparse.ArgumentParser(description="analyse a meg dataset and"
                                                 "find faulty and/or unphysical"
                                                 "measurements")

    parser.add_argument("--filename", required=True, type=str, help="filename of the dataset to analyse")
    parser.add_argument('--filters', nargs='+', choices=["uniq", "segment", "gradient"],
                        default=["uniq", "segment", "gradient"], help="the basic filters to use")
    parser.add_argument("-p", "--physicality", action="store_true", default=False, help="do physicality analysis")
    return parser.parse_args()


def secondver():
    # fname = "sample_data02.npz"
    fname = "sample_data24.npz"
    channels = ["MEG2*1"]
    signals, names, time, n_chan = fr.get_signals(fname, channels=channels)

    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(signals, names, n_chan)
    hf.plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, exec_times)


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


def thirdver():
    args = arg_parser()
    detecs = np.load("array120_trans_newnames.npz")

    print("analysing " + args.filename)
    print()

    signals, names, t, n_chan = fr.get_signals(args.filename)
    print("beginning analysis with the following filters:", args.filters)
    print()
    start_time = time.time()
    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(signals, names, n_chan, filters=args.filters)
    end_time = time.time()
    filt_time = (end_time - start_time) / 60
    print("time elapsed in filtering: " + str(filt_time) + " mins")
    print()

    if args.physicality:
        print("-----------------------------------------------------")
        print("beginning physicality analysis")
        print()
        start_time = time.time()
        all_diffs, all_rel_diffs, chan_dict = sa.check_all_phys(signals, detecs, names, n_chan, bad_segs,
                                                                ave_window=100, ave_sens=10 ** (-12))

        status, confidence = sa.analyse_phys_dat(all_diffs, names, all_rel_diffs, chan_dict)
        end_time = time.time()
        phys_time = (end_time - start_time) / 60
        print()
        print("time elapsed in physicality analysis: " + str(phys_time) + " mins")
    else:
        status = []
        confidence = []
        all_rel_diffs = []
        chan_dict = []
        phys_time = 0

    tot_time = phys_time + filt_time
    print("-----------------------------------------------------")
    print()
    print_results(names, signals, signal_statuses, bad_segs, suspicious_segs,
                  status, confidence, all_rel_diffs, chan_dict)
    print("total time elapsed: " + str(tot_time) + " mins")


if __name__ == '__main__':
    #tf.test_new_excluder()
    thirdver()


# fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
#
# detec_quivers = []
#
# for i in range(len(near_vs)):
#     print(nearby_names[i])
#     r = near_rs[i]
#     v = near_vs[i]
#     detec_quivers.append(ax.quiver(r[0], r[1], r[2], v[0], v[1], v[2], length=.01, color="black"))
#
# first_mag = magnus[0]
# print(first_mag)
# len_scale = 10 ** 20
# mag_len = np.linalg.norm(first_mag) * len_scale
# quiver = ax.quiver(comp_r[0], comp_r[1], comp_r[2], first_mag[0], first_mag[1], first_mag[2], length=mag_len,
#                    color="red")
# print(quiver)
#
#
# def update(ani_i):
#     global quiver
#     print(ani_i)
#     new_mag = magnus[ani_i]
#     print(new_mag)
#     mag_len = np.linalg.norm(new_mag) * len_scale
#     quiver.remove()
#     quiver = ax.quiver(comp_r[0], comp_r[1], comp_r[2], new_mag[0], new_mag[1], new_mag[2], length=mag_len, color="red")
#
#
# from matplotlib.animation import FuncAnimation
#
# ani = FuncAnimation(fig, update, frames=range(frames), repeat=False)
#
# plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
