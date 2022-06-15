import signal_generator as sg
import signal_analysis as sa
import matplotlib.pyplot as plt
import file_reader as fr
import numpy as np
import pandas as pd

methods = ["Pelt", "Dynp", "Binseg", "Window"]
datadir = "example_data_for_patrik/"

def reorganize_signal(signal):
    size = len(signal)
    new_sig = np.zeros(size)
    print(size)

    for i in range(len(signal)):
        new_sig[i] = signal[i][0]

    return new_sig

def dataload():
    fname = datadir + "many_failed.npz"
    data = fr.load_all(fname)
    chan_name = "MEG1921"
    data = data.clip((0.210, 0.50))
    signal = data.subpool([chan_name]).data
    time = data.subpool([chan_name]).time

    #print(len(time))

    signal = reorganize_signal(signal)
    method = methods[3]



    dsig = np.gradient(signal)

    points = sa.find_changes(dsig, method)
    sa.rpt_plot(signal, points)
    plt.title(method)

    #fig, (ax1, ax2) = plt.subplots(2, 1)
    #ax1.plot(time, signal)

    #ax2.plot(time, dsig)

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
        points = sa.find_changes(signal, method)
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


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
