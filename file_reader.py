import dataio as dat

datadir = "example_data_for_patrik/"

def find_signals(channels, signals, names):
    indices = []

    for channel in channels:
        i = names.index(channel)
        indices.append(i)

    #print(indices)

    signals_to_return = []

    for index in indices:
        signals_to_return.append(signals[index])

    return signals_to_return

def reorganize_signals(signals, n):
    new_signals = []
    for i in range(n):
        signal = signals[:, i]
        new_signals.append(signal)

    return new_signals

def load_all(fname):
    return dat.MultiChannelData.load_npz(fname)

def get_signals(fname, channels=["MEG*1", "MEG*4"]):
    fname_full = datadir + fname
    data = load_all(fname_full).subpool(channels).clip((0.210, 0.50))
    unorganized_signals = data.data
    names = data.names
    n_chan = data.n_channels
    time = data.time
    signals = reorganize_signals(unorganized_signals, n_chan)

    return signals, names, time, n_chan