from scipy.signal import firls

def hpfilter_fir(Fstop, Fpass, fs, Ntaps):
    """
    FIR least-squares highpass filter design using the FIRLS function.

    Args:
        Fstop, Fband : Frequency (in Hz) used for determining frequency bands
        fs           : Sampling frequency
        Ntaps        : Number of filter taps
    """

    return firls(Ntaps, [0, Fstop, Fpass, fs/2], desired=[0,0,1,1], weight=[1, 1], fs=fs)