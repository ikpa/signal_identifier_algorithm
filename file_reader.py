import dataio as dat

time_window = (0.210, 0.50)

# finds and returns signals based on channel names
def find_signals(channels, signals, names):
    indices = []

    for channel in channels:
        i = names.index(channel)
        indices.append(i)

    signals_to_return = []

    for index in indices:
        signals_to_return.append(signals[index])

    return signals_to_return


# reformats signals so that a single array contains one signal instead of
# one array containing one point in time
def reorganize_signals(signals, n):
    new_signals = []
    for i in range(n):
        signal = signals[:, i]
        new_signals.append(signal)

    return new_signals


# load all data from file
def load_all(fname):
    return dat.MultiChannelData.load_npz(fname)


# get all reformatted signals from filename
def get_signals(fname, channels=["MEG*1", "MEG*4"], time_win=time_window):
    fname_full = fname
    data = load_all(fname_full).subpool(channels).clip(time_win)
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = reorganize_signals(unorganized_signals, n_chan)

    return signals, names, time, n_chan
