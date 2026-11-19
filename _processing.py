import numpy as np
from numpy.fft import rfft
import mne

RANDOM_STATE = 5

freq_bands = {
    "delta" : (0.5, 4),
    "theta" : (4, 8),
    "alpha" : (8, 13),
    "beta" : (13, 30),
    "gamma" : (30, 45),
    "all" : (0.5, 45)
}

def extract_band(eeg : np.ndarray, band : str, Fs = 500):
    fmin = freq_bands[band][0]
    fmax = freq_bands[band][1]

    filtered =  mne.filter.filter_data(eeg, sfreq=Fs, l_freq=fmin, h_freq=fmax,
                                       method = "fir", fir_window = "hann",
                                       verbose=False)
    return filtered

def decimate(signal: np.ndarray, target_Fs = 100, Fs = 500):
    w_size = int(np.floor(Fs / target_Fs))
    N_new = int(signal.shape[1] // w_size)
    new = np.zeros((signal.shape[0], N_new))

    for i in range(N_new-1):
        partial = signal[:, i*w_size : (i+1) * w_size]
        new[:, i] = np.mean(partial, axis = 1)

    return new

def surrogate(signal : np.ndarray, n = 10):
    rng = np.random.default_rng()
    SIGNAL = rfft(signal, axis=1)
    magnitude = np.abs(SIGNAL)
    res = np.zeros((n, len(signal), signal.shape[1]))
    
    for s in range(n):
        random_phases = rng.uniform(
            -np.pi, np.pi, size=SIGNAL.shape
        )
        random_phases[:, 0] = 0
        # Nyquist component must also remain real for even-length signals
        if signal.shape[1] % 2 == 0:
            random_phases[:, -1] = 0

        
        randomized_fft = magnitude * np.exp(1j * random_phases)
        surrogate = np.fft.irfft(
            randomized_fft,
            axis=1
        )

        res[s] = surrogate

    return res