import numpy as np

from librosa import lpc
from scipy.signal import lfilter, hanning

from adles.filters.hpfilter_fir import hpfilter_fir

def gfmiaif(frame, fs, p_vt, p_gl, d, hpfilt,
            hpfilt_in=np.zeros(1)):
    """
        This function estimates the linear prediction coefficients of both
        vocal tract and glottis filters from a speech signal frame with the
        GFM-IAIF method [1].
        The latter is an extension of IAIF [2], with an improved pre-emphasis
        step, that allows to extract a wide-band glottis response,
        incorporating both glottal formant and spectral tilt characteristics.
        This function is based on the iaif.m implementation from the COVAREP
        toolbox [3].

        Args:

            frame     : Speech signal frame (of shape Nx1)
            fs        : Sampling Frequency
            p_vt        : Order of LP analysis for vocal tract (default 48)
            p_gl        : Order of LP analysis for glottal source (default 3)
            d         : Leaky integration co-efficient (default 0.99)
            hpfilt    : High-pass filter flag (0: do not apply, 1...N: apply N times)
            win       : Window used before LPC (default Hanning)
            hpfilt_in : Pass a precomputed FIRLS filter in if it's alread

        Returns:

            g         : Glottal volume velocity waveform
            dg        : Glottal volume velocity derivative waveform
            a         : LPC coefficients of vocal tract
            ag        : LPC coefficients of source spectrum
    """

    x = frame
    preflt = p_vt

    # High pass filter speech in order to remove possible low 
    # frequency fluctuations (Linear-phase FIR, Fc = 70Hz)
    if hpfilt > 0:
        Fstop = 40               # Stopband frequency
        Fpass = 70               # Passband frequency
        Nfir = int(300/16000*fs) # FIR numerator order

        if Nfir % 2 == 0:
            Nfir += 1

        # it is very very expensive to calculate the firls filter! However, as 
        # long as the fs does not change, the firls filter does not change.
        # Therefore, the computed filter is returned and can be passed to this
        # function later on to avoid the calculated of the (same) filter. 
        if hpfilt_in.shape[0] != 1:
            B = hpfilt_in
        else:
            B = hpfilter_fir(Fstop, Fpass, fs, Nfir)   

        hpfilter_out = B

        for i in range(hpfilt):
            x = lfilter(B, np.ones(1), np.hstack((x, np.zeros(int(B.shape[0]/2-1)))))
            x = x[int(B.shape[0]/2)-1:]

    # Estimate the combined effect of the glottal flow and the lip radiation
    # (Hg1) and cancel it out through inverse filtering. Note that before
    # filtering, a mean-normalized pre-frame ramp is appended in order to
    # diminish ripple in the beginning of the frame. The ramp is removed after
    # filtering.

    if x.shape[0] > p_vt:
        win = hanning(x.shape[0])
        signal = np.hstack((np.linspace(-x[0], x[0], preflt), x))
        idx = np.arange(preflt,signal.shape[0])

        Hg1 = lpc(np.multiply(x, win),1)
        y = lfilter(Hg1, np.ones(1), signal)
        y = y[idx]

        # Estimate the effect of the vocal tract (Hvt1) and cancel it out through
        # inverse filtering. The effect of the lip radiation is canceled through
        # integration. Signal g1 is the first estimate of the glottal flow.
        Hvt1 = lpc(np.multiply(y, win), p_vt)
        g1 = lfilter(Hvt1, np.ones(1), signal)
        g1 = lfilter(np.ones(1), np.array([1, -d]), g1)
        g1 = g1[idx]

        # Re-estimate the effect of the glottal flow (Hg2). Cancel the contribution
        # of the glottis and the lip radiation through inverse filtering and
        # integration, respectively.
        Hg2 = lpc(np.multiply(g1, win), p_gl)
        y = lfilter(Hg2, np.ones(1), signal)
        y = lfilter(np.ones(1), np.array([1, -d]), y)
        y = y[idx]

        # Estimate the model for the vocal tract (Hvt2) and cancel it out through
        # inverse filtering. The final estimate of the glottal flow is obtained
        # through canceling the effect of the lip radiation.
        Hvt2 = lpc(np.multiply(y, win), p_vt)
        dg = lfilter(Hvt2, np.ones(1), signal)
        g = lfilter(np.ones(1), np.array([1, -d]), dg)
        g = g[preflt:]
        dg = dg[idx]

        # Set vocal tract model to 'a' and glottal source spectral model to 'ag'
        a = Hvt2
        ag = Hg2

        return g, dg, a, ag, hpfilter_out

    else:
        return [], [], [], [], None