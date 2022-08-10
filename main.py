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
    # fname = "sample_data02.npz"
    fname = "sample_data24.npz"
    channels = ["MEG2*1"]
    signals, names, time, n_chan = fr.get_signals(fname, channels=channels)

    signal_statuses, bad_segs, suspicious_segs, exec_times = sa.analyse_all_neo(signals, names, n_chan)
    hf.plot_in_order_ver3(signals, names, n_chan, signal_statuses, bad_segs, suspicious_segs, exec_times)


# if __name__ == '__main__':
# basic()
# analysis()
# dataload()
# averagetest()
# firstver()
# secondver()
# plottest()
# animtest()
# simo()
# nearby()
# names()
# simulation()
# test_uniq()
# overlap()
# test_hz()
# tf.compare_nearby2()
# tf.animate_vectors()
# tf.detrend_grad()
# tf.test_magn()
# vector_closeness()
# angle_test()
# signal_sim()

smooth_window = 401
offset = int(smooth_window / 2)


def filter_and_smooth(signal):
    filter_i = sa.filter_start(signal)
    filtered_signal = signal[filter_i:]
    x = list(range(filter_i, len(signal)))

    smooth_signal = sa.smooth(filtered_signal, window_len=smooth_window)
    smooth_x = [x - offset + filter_i for x in list(range(len(smooth_signal)))]
    new_smooth = []
    for i in range(len(filtered_signal)):
        new_smooth.append(smooth_signal[i + offset])

    return filtered_signal, x, smooth_signal, smooth_x, new_smooth

fname = "many_many_successful.npz"
signals, names, time, n_chan = fr.get_signals(fname)

detecs = np.load("array120_trans_newnames.npz")

comp_detec = "MEG1331"
#comp_detec = "MEG0711"
comp_sig = fr.find_signals([comp_detec], signals, names)[0]
comp_v = detecs[comp_detec][:3, 2]
comp_r = detecs[comp_detec][:3, 3]

nearby_names = sa.find_nearby_detectors(comp_detec, detecs)
near_sigs = fr.find_signals(nearby_names, signals, names)

near_vs = []
near_rs = []

for name in nearby_names:
    near_vs.append(detecs[name][:3, 2])
    near_rs.append(detecs[name][:3, 3])

nearby_names.append(comp_detec)
near_sigs.append(comp_sig)
near_vs.append(comp_v)
near_rs.append(comp_r)

filtered_sigs = []
xs = []

for signal in near_sigs:
    filtered_signal, x, smooth_signal, smooth_x, new_smooth = filter_and_smooth(signal)
    filtered_sigs.append(np.gradient(new_smooth))
    xs.append(x)

magnus, mag_is, cropped_sigs, new_x = sa.calc_magn_field_from_signals(filtered_sigs, xs, near_vs, ave_window=10)

reconst_sigs = []

for i in range(len(near_rs)):
    r = near_rs[i]
    v = near_vs[i]
    reconst_sig = []

    for j in range(len(magnus)):
        magn = magnus[j]
        reconst_sig.append(np.dot(magn, v))

    reconst_sigs.append(reconst_sig)

frames = len(magnus)
signal_len = len(new_x)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

colors = plt.cm.rainbow(np.linspace(0, 1, len(near_sigs)))
for i in range(len(near_sigs)):
    color = colors[i]
    plot_name = nearby_names[i]
    ax1.plot(new_x, cropped_sigs[i], color=color, label=plot_name)
    ax2.plot(mag_is, reconst_sigs[i], color=color, label=plot_name)

ax2.legend()
plt.show()

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

detec_quivers = []

for i in range(len(near_vs)):
    print(nearby_names[i])
    r = near_rs[i]
    v = near_vs[i]
    detec_quivers.append(ax.quiver(r[0], r[1], r[2], v[0], v[1], v[2], length=.01, color="black"))

first_mag = magnus[0]
print(first_mag)
len_scale = 10 ** 20
mag_len = np.linalg.norm(first_mag) * len_scale
quiver = ax.quiver(comp_r[0], comp_r[1], comp_r[2], first_mag[0], first_mag[1], first_mag[2], length=mag_len,
                   color="red")
print(quiver)


def update(ani_i):
    global quiver
    print(ani_i)
    print(quiver)
    new_mag = magnus[ani_i]
    print(new_mag)
    mag_len = np.linalg.norm(new_mag) * len_scale
    quiver.remove()
    quiver = ax.quiver(comp_r[0], comp_r[1], comp_r[2], new_mag[0], new_mag[1], new_mag[2], length=mag_len, color="red")


from matplotlib.animation import FuncAnimation

ani = FuncAnimation(fig, update, frames=range(frames), repeat=False)

plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
