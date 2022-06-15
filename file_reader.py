import dataio as dat

def load_all(fname):
    return dat.MultiChannelData.load_npz(fname)
