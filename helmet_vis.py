import calc_eddy_currents_for_patrik.viz as viz
import mayavi as may
import PyQt5

def plot_all(names, datas):
    viz.plot_sensor_data(names, datas, cmap="Greys")
    may.mlab.show()
