import dataio as dat
from helper_funcs import reorganize_signals

time_window = (0.210, 0.50)

def write_data_compact(fname, data, col_names, divider=";\t"):
    f = open(fname, "w")
    header = ""

    for name in col_names:
        if header == "":
            header += name
        else:
            header += divider + name

    f.write(header + "\n")

    n_points = len(data[0])

    for i in range(n_points):
        line = ""

        for data_arr in data:
            data_value = data_arr[i]
            if line == "":
                line += str(data_value)
            else:
                line += divider + str(data_value)

        f.write(line + "\n")

    f.close()

# load all data from file
def load_all(fname):
    return dat.MultiChannelData.load_npz(fname)

# get all reformatted signals from filename
def get_signals(fname, channels=["MEG*1", "MEG*4"]):

    fname_full = fname
    data = load_all(fname_full).subpool(channels).clip(time_window)
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = reorganize_signals(unorganized_signals, n_chan)

    return signals, names, time, n_chan

class Printer:
    def __init__(self, write_mode, file=None):
        self.mode = write_mode
        self.f = file

    def extended_write(self, *args, additional_mode=""):
        mode = self.mode + additional_mode
        f = self.f

        if mode == "none":
            return

        txt = ""

        for i in range(len(args)):
            arg = args[i]
            if not isinstance(arg, str):
                arg = str(arg)

            if i == 0:
                txt += arg
            else:
                txt += " " + arg


        if "print" in mode:
            print(txt)

        if "file" in mode and f is not None:
            f.write(txt + "\n")
