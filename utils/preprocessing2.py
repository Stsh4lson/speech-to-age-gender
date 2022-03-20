import numpy as np
import librosa

class EigenDecomposition:
    def __init__(self, fs=48000, nfft=512, freq_range=[200, 12200]):
        self.fs = fs
        self.nfft = nfft
        self.freq_range = [int(np.round(f / self.fs * self.nfft)) for f in freq_range]
        self.freq_bins = np.arange(self.freq_range[0], self.freq_range[1], dtype=int)
        
    def stft(self, X):
        S = librosa.core.stft(np.array(X), n_fft=self.nfft)
        return S

    def compute_correlation_matricesvec(self, X):
        # select frequency bins
        X = X[list(self.freq_bins), :].T
        # Compute PSD
        C_hat = np.matmul(X[..., None], np.conjugate(X[..., None, :]))
        # Average over time-frames
        C_hat = np.mean(C_hat, axis=0)
        return C_hat
    
    def subspace_decomposition(self, R):
        # eigenvalue decomposition
        # This method is specialized for Hermitian symmetric matrices,
        # which is the case since R is a covariance matrix
        w, v = np.linalg.eigh(R)
        v = np.abs(v)
        return v
    
    def process(self, X):
        X = self.stft(X)
        X = self.compute_correlation_matricesvec(X)
        X = self.subspace_decomposition(X)
        return X