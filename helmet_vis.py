import calc_eddy_currents_for_patrik.viz as viz
from mayavi import mlab
import numpy as np

def get_single_point(signals, i, n=1):
    points = []

    for signal in signals:
        point = signal[i]
        for j in range(n):
            points.append(point)

    return points

def helmet_animation(names, signals, frames, bads=[]):
    signal_len = len(signals[0])
    min = np.amin(signals)
    max = np.amax(signals)

    points = get_single_point(signals, 0)

    s = plot_all(names, points, bads=bads, cmap="PiYG",
                     vmax=max, vmin=min, plot=False)
    text = mlab.text(0.85, 0.125, "0.00", width=0.09)
    # print(s.mlab_source.scalars)
    # print(points)

    @mlab.animate(delay=50)
    def anim():
        for i in range(1, frames):
            j = int(signal_len * (i / frames))
            text.set(text=str(round(j/signal_len * 100)) + "%")
            print("j= ", j)

            points = get_single_point(signals, j, n=4)
            s.mlab_source.scalars = points
            yield

    anim()
    mlab.show()

def plot_all(names, datas, bads=[], cmap="Greys", vmin = None, vmax = None,
             plot=True):
    s = viz.plot_sensor_data(names, datas, cmap=cmap, bads=bads,
                         vmin=vmin, vmax=vmax)
    mlab.colorbar()

    if plot:
        mlab.show()
    return s
