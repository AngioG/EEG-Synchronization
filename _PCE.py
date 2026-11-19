import numpy as np
from _processing import extract_band, decimate

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

def short_term_PCE(eeg : np.ndarray, Fs = 500,
                   window_size : float = 0, overlap = 0,
                   do_decimate = False) -> list[float]:
    if do_decimate:
        eeg = decimate(eeg)
        Fs = 100
    if window_size <= 0 or window_size * Fs > eeg.shape[1]:
        step = int(eeg.shape[1])
    elif overlap == 0 or overlap > window_size:
        step = int(np.floor(window_size * Fs))
    else:
        step = int(np.floor((window_size - overlap) * Fs))
    n = int(np.floor(eeg.shape[1] // step))
    pce = []
    for i in range(n):
        start = int(i * step)
        windowed = eeg[:, start:start+step]
        pce.append(PCE(windowed)[0])
    return pce

def PCE_by_band(eeg : np.ndarray, bands : list[str] = list(freq_bands.keys()),
                Fs = 500, window_size : float = 0, do_decimate = False):
    if do_decimate:
        eeg = decimate(eeg)
        Fs = 100
    PCEs = []
    for name in bands:
        filtered = extract_band(eeg, name, Fs=Fs)
        if window_size > 0:
            pce = np.mean(short_term_PCE(filtered, Fs, window_size))
        else: 
            pce, _ = PCE(filtered)
        PCEs.append({
            "band" : name,
            "PCE" : pce
        })

    return PCEs