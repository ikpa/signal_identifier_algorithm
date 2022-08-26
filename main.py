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


if __name__ == '__main__':
    tf.test_new_excluder()


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
