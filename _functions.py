import numpy as np
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

def PCE (W):
    Cov = np.cov(W)
    eigenvalues, _ = np.linalg.eigh(Cov)
    M = len(eigenvalues)
    eigenvalues_norm = eigenvalues / sum(eigenvalues)
    
    PCE = - np.sum(eigenvalues_norm * np.log2(eigenvalues_norm)) / np.log2(len(eigenvalues_norm))
    
    return PCE, eigenvalues_norm

def extract_band(eeg : np.ndarray, band : str, Fs = 500):
    fmin = freq_bands[band][0]
    fmax = freq_bands[band][1]

    filtered =  mne.filter.filter_data(eeg, sfreq=Fs, l_freq=fmin, h_freq=fmax,
                                       method = "fir", fir_window = "hann",
                                       verbose=False)
    return filtered

def PCE_by_band(eeg : np.ndarray, bands : list[str] = list(freq_bands.keys()), Fs = 500):
    PCEs = []
    for name in bands:
        filtered = extract_band(eeg, name, Fs=Fs)
        pce, _ = PCE(filtered)
        PCEs.append({
            "band" : name,
            "PCE" : pce
        })

    return PCEs

def pooled_std(x : np.ndarray, y : np.ndarray) -> float:
    var_x = np.var(x)
    var_y = np.var(y)
    n_x = len(x)
    n_y = len(y)

    return np.sqrt(
        ((n_x - 1) * var_x + (n_y - 1) * var_y)
        /
        (n_x + n_y - 2)
    )

def cohen_d(x : np.ndarray, y : np.ndarray) -> float:
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    s = pooled_std(x, y)

    return (mean_x - mean_y) / s