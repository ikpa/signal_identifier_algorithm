import numpy as np
import mayavi.mlab as mlab
#import helper_funcs as hf
import viz as viz


# def helmet_animation(names, signals, frames=1000, bads=[], arrows=False, cmap="PiYG",
#                      vlims=[]):
#     signal_len = len(signals[0])
#     # mini = np.amin(signals)
#     # maxi = np.amax(signals)
#
#     if len(vlims) == 0:
#         mini, maxi = hf.find_min_max_ragged(signals)
#     else:
#         mini = vlims[0]
#         maxi = vlims[1]
#
#     points = hf.get_single_point(signals, 0)
#
#     s, a = plot_all(names, points, bads=bads, cmap=cmap,
#                     vmax=maxi, vmin=mini, plot=False, arrows=arrows)
#     text = mlab.text(0.85, 0.125, "0.00", width=0.09)
#
#     # print(s.mlab_source.scalars)
#     # print(points)
#
#     @mlab.animate(delay=50)
#     def anim():
#         for i in range(1, frames):
#             j = int(signal_len * (i / frames))
#             text.set(text=str(round(j / signal_len * 100)) + "%")
#             print("j= ", j)
#
#             points = hf.get_single_point(signals, j, n=4)
#             s.mlab_source.scalars = points
#
#             if a != None:
#                 print(a.mlab_source.x)
#
#             yield
#
#     anim()
#     mlab.show()


def plot_all(names, datas, bads=[], cmap="Reds", vmin=None, vmax=None,
             plot=True, arrows=False):
    s = viz.plot_sensor_data(names, datas, cmap=cmap, bads=bads,
                             vmin=vmin, vmax=vmax)
    #mlab.colorbar()

    if arrows:
        detecs = np.load("array120_trans_newnames.npz")

        rs = []
        vecs = []

        i = 0
        for detector in detecs:
            d = detecs[detector]
            print(d)
            r = d[:3, 3]
            vec = d[:3, 2]

            if i > len(datas) - 1:
                length = 0
            else:
                length = datas[i]

            vec = [x * length for x in vec]

            rs.append(r)
            vecs.append(vec)
            i += 1

        #a = mlab.quiver3d(rs[0], rs[1], rs[2], vecs[0], vecs[1], vecs[2])
    else:
        a = None

    if plot:
        mlab.show()
    return s, a
