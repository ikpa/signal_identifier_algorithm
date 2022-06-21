import signal_generator as sg
import signal_analysis as sa
import matplotlib.pyplot as plt
import file_reader as fr
import numpy as np
import pandas as pd

methods = ["Pelt", "Dynp", "Binseg", "Window"]
datadir = "example_data_for_patrik/"

def firstver():
    fname = datadir + "many_many_successful2.npz"
    data = fr.load_all(fname).subpool(["MEG*1"]).clip((0.210, 0.50))
    time = data.time
    statuses, fracs, uniq_stats_list, exec_times = sa.analyse_all(data)
    n = data.n_channels

    for i in range(n):
        name = data.names[i]
        signal = data.data[:, i]
        bad = statuses[i]
        frac = fracs[i]
        uniq_stats = uniq_stats_list[i]
        exec_time = exec_times[i]

        print(name)
        if bad[0]:
            #print("bad, skipping")
            #print()
            #continue
            status = "bad"
        else:
            status = "good"

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.plot(signal)
        ax1.axvline(x=sa.filter_start(signal), linestyle="--")
        ax2.plot(np.gradient(signal))
        title = name + ": " + status
        if not len(uniq_stats) == 0:

            print(uniq_stats[0])
            if not len(uniq_stats[0]) == 0:
                same_sum = uniq_stats[2][len(uniq_stats[2]) - 1] - uniq_stats[1][0]
                same_frac = same_sum / len(signal)
                for j in range(len(uniq_stats[0])):
                    # plt.axvline(x=start_is[i], linestyle="--", color="black")
                    # plt.axvline(x=end_is[i], linestyle="--", color="black")
                    ax1.axvspan(uniq_stats[1][j], uniq_stats[2][j], alpha=.5)
                title += ", confidence: " + str(bad[1])

        plt.title(title)


        plt.grid()

        plt.show()

        print()

def averagetest():
    fname = datadir + "many_failed.npz"
    channame = "MEG0631"
    channame = "MEG1421"
    channame = "MEG1731"
    data = fr.load_all(fname).subpool([channame]).clip((0.210, 0.50))
    signal = sa.reorganize_signal(data.data)
    print(signal)
    uniq_stats, bad = sa.segment_filter(signal)
    averages = sa.find_start_of_seg(signal, uniq_stats[1][0])

    print(len(signal))
    print(len(averages))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex= True)
    ax1.plot(signal)
    ax2.plot(averages)
    ax3.plot(np.gradient(signal))

    plt.show()

def dataload():
    fname = datadir + "many_failed.npz"
    data = fr.load_all(fname).subpool(["MEG*1"]).clip((0.210, 0.50))
    names = data.names
    n = data.n_channels
    #time = data.subpool([chan_name]).time

    for j in range(n):
        signal = data.data[:, j]
        #vals, lens, start_is, end_is = sa.find_uniq_segments(signal, 4)
        uniq_stats = sa.segment_filter(signal, 4)
        print(uniq_stats[0])
        print(uniq_stats[1])
        print()
        #print(len(lens))
        #print(len(start_is))

        #print(len(np.unique(signal)))

        plt.plot(signal, ".-")
        plt.title(names[j])
        for i in range(len(uniq_stats[0])):
            # plt.axvline(x=start_is[i], linestyle="--", color="black")
            # plt.axvline(x=end_is[i], linestyle="--", color="black")
            plt.axvspan(uniq_stats[2][i], uniq_stats[3][i], alpha=.5)

        plt.show()


def analysis():
    window = 10
    signal, t, i = sg.gen_signal(containsError=True, containsNoise=True)
    #i = i - (window - 1)
    #print(type(signal))
    #print(len(signal))
    #signal = pd.Series(signal).rolling(window=window).mean()
    #signal = signal[~np.isnan(signal)]
    #print(len(signal))
    #dsig = np.gradient(signal)
    #print(dsig)
    #print(type(signal))
    print()
    print("actual breakpoint", i)

    for method in methods:
        print("calculating with method " + method)
        points = sa.find_changes_rpt(signal, method)
        sa.rpt_plot(signal, points)
        plt.title(method)
        print("predicted breakpoint(s)", points)


    #sa.rpt_plot(dsig, points, [i])
    plt.show()

def basic():
    signal, t, i = sg.gen_signal(containsNoise=True, containsError=True)
    signal_smooth = pd.Series(signal).rolling(window=10).mean()
    dsig = np.gradient(np.gradient(signal))
    dsig_smooth = pd.Series(dsig).rolling(window=7).mean()
    # dsig_smooth = np.gradient(signal_smooth)

    error_t = t[i]
    # print(error_t)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t, signal, label="normal")
    ax1.plot(t, signal_smooth, label="filtered")

    ax2.plot(t, dsig, label="normal")
    ax2.plot(t, dsig_smooth, label="filtered")
    if i != None:
        ax1.axvline(x=error_t, linestyle="--", c="black")
        ax2.axvline(x=error_t, linestyle="--", c="black")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    #basic()
    #analysis()
    #dataload()
    #averagetest()
    firstver()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
