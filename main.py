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

methods = ["Pelt", "Dynp", "Binseg", "Window"]
datadir = "example_data_for_patrik/"

def secondver():
    #fname = "sample_data02.npz"
    fname = "sample_data24.npz"
    channels = ["MEG2*1"]
    signals, names, time, n_chan = fr.get_signals(fname, channels=channels)

    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(signals, names, n_chan)
    hf.plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, exec_times)


if __name__ == '__main__':
    # basic()
    # analysis()
    # dataload()
    # averagetest()
    # firstver()
    #secondver()
    # plottest()
    # animtest()
    # simo()
    # nearby()
    # names()
    # simulation()
    # test_uniq()
    # overlap()
    #test_hz()
    #tf.compare_nearby2()
    #tf.animate_vectors()
    tf.detrend_grad()
    #vector_closeness()
    #angle_test()
    #signal_sim()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
