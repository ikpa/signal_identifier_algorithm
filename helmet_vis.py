import calc_eddy_currents_for_patrik.viz as viz
import mayavi as may

def plot_all(names, datas, cmap="Greys"):
    viz.plot_sensor_data(names, datas, cmap=cmap)
    may.mlab.show()
