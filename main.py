import signal_generator as sg
import signal_analysis as sa
import matplotlib.pyplot as plt
import file_reader as fr
import numpy as np
import pandas as pd

methods = ["Pelt", "Dynp", "Binseg", "Window"]
datadir = "example_data_for_patrik/"

def firstver():
    fname = datadir + "many_failed.npz"
    data = fr.load_all(fname).subpool(["MEG*1"]).clip((0.210, 0.50))
    time = data.time
    categories, fracs = sa.analyse_all(data)
    n = data.n_channels

    print(np.shape(categories))

    for i in range(n):
        name = data.names[i]
        signal = data.data[:, i]
        category = categories[i]
        frac = fracs[i]
        print(category)

        plt.figure()
        plt.plot(time, signal)
        if category == 0:
            status = "bad"

        if category == 1:
            status = "ambiguous"

        if category == 2:
            status = "good"

        plt.title(name + ": " + status + ", " + str(frac))
        plt.grid()
        plt.show()

def dataload():
    fname = datadir + "many_successful.npz"
    data = fr.load_all(fname).subpool(["MEG*1"]).clip((0.210, 0.50))
    names = data.names
    n = data.n_channels
    #time = data.subpool([chan_name]).time

    for j in range(n):
        signal = data.data[:, j]
        vals, lens, start_is, end_is = sa.find_uniq_segments(signal, 4)
        print(vals)
        print(lens)
        print()
        #print(len(lens))
        #print(len(start_is))

        #print(len(np.unique(signal)))

        plt.plot(signal, ".-")
        plt.title(names[j])
        for i in range(len(start_is)):
            # plt.axvline(x=start_is[i], linestyle="--", color="black")
            # plt.axvline(x=end_is[i], linestyle="--", color="black")
            plt.axvspan(start_is[i], end_is[i], alpha=.5)

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
    dataload()
    #firstver()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
