import numpy as np
from sklearn.decomposition import PCA 

RANDOM_STATE = 5

freq_bands = {
    "delta" : 1,
    "theta" : 4,
    "alpha" : 8,
    "beta" : 30,
    "gamma" : 45
}

def PCE (W, bands : list[str] = list(freq_bands.keys())):
    Cov = np.cov(W)
    eigenvalues, _ = np.linalg.eigh(Cov)
    M = len(eigenvalues)
    eigenvalues_norm = eigenvalues / sum(eigenvalues)
    
    PCE = - np.sum(eigenvalues_norm * np.log2(eigenvalues_norm)) / np.log2(len(eigenvalues_norm))
    
    return PCE, eigenvalues_norm