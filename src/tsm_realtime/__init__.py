"""
Realtime Time-Stretch Methods
Provides realtime hybrid audio time-stretching capabilities using
Phase Vocoder (PV) and Overlap-Add (OLA) techniques. It implements both baseline
and lookup-based approaches for harmonic/percussive source separation.

Author: Sayema Lubis, Jared Carreno
"""

from typing import Optional

import numpy as np
import scipy
import librosa as lb
import pyaudio
from pynput import keyboard
import threading


# ===== Global alpha =====
alpha = 1.0

# Private functions
# --------------------------------
def _float2pcm(sig, dtype='int16'):
    """Convert floating point signal to PCM format.
    
    Args:
        sig (np.ndarray): Floating point input signal (range [-1, 1])
        dtype (str): Target PCM format (default: 'int16')
        
    Returns:
        np.ndarray: PCM signal in specified format
    """
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

def _calc_sum_squared_window(window, hop_length):
    """Calculate the denominator term for PV synthesis normalization.
    
    This function computes the sum of squared window functions shifted by
    hop_length increments, which is used for proper normalization in
    Phase Vocoder synthesis.
    
    Args:
        window (np.ndarray): Analysis window function
        hop_length (int): Hop size in samples (must divide window length)
        
    Returns:
        np.ndarray: Normalization factor array
        
    Raises:
        AssertionError: If hop_length doesn't divide window length evenly
    """
    assert (len(window) % hop_length == 0), "Hop length does not divide the window evenly."
    numShifts = len(window) // hop_length
    den = np.zeros_like(window)
    for i in range(numShifts):
        den += np.roll(np.square(window), i * hop_length)
    return den

def _invert_stft(S, hop_length, window):
    """Reconstruct a signal from a modified STFT matrix.
    
    This function performs inverse STFT reconstruction using overlap-add
    synthesis with proper window normalization.
    
    Args:
        S (np.ndarray): Modified STFT matrix (complex, shape: [k_bins, n_frames])
        hop_length (int): Synthesis hop size in samples
        window (np.ndarray): Analysis window function
        
    Returns:
        np.ndarray: Reconstructed time-domain signal
    """
    L = len(window)
    fft_size = (S.shape[0] - 1) * 2
    Sfull = np.zeros((fft_size, S.shape[1]), dtype=np.complex64)
    Sfull[0:S.shape[0], :] = S
    Sfull[S.shape[0]:, :] = np.conj(np.flipud(S[1:fft_size // 2, :]))
    frames = np.zeros_like(Sfull)
    for i in range(frames.shape[1]):
        frames[:, i] = np.fft.ifft(Sfull[:, i])
    frames = np.real(frames)
    den = _calc_sum_squared_window(window, hop_length)
    frames = frames * window.reshape((-1, 1)) / den.reshape((-1, 1))
    y = np.zeros(hop_length * (frames.shape[1] - 1) + L)
    for i in range(frames.shape[1]):
        offset = i * hop_length
        y[offset:offset + L] += frames[:, i]
    return y

def _harmonic_percussive_separation(x, sr=22050, fft_size=2048, hop_length=512, lh=6, lp=6):
    """Perform Harmonic/Percussive source separation using median filtering.
    
    This function separates an audio signal into harmonic and percussive components
    using median filtering in the time-frequency domain.
    
    Args:
        x (np.ndarray): Input audio signal
        sr (int): Sampling rate (default: 22050)
        fft_size (int): Size of the FFT window (default: 2048)
        hop_length (int): Hop size for STFT (default: 512)
        lh (int): Half-length of median filter for harmonic components (default: 6)
        lp (int): Half-length of median filter for percussive components (default: 6)
        
    Returns:
        tuple: (xh, xp) where xh is harmonic component and xp is percussive component
    """
    window = np.hanning(fft_size)
    X = lb.core.stft(x, n_fft=fft_size, hop_length=hop_length, window=window, center=False)
    Y = np.abs(X)
    Yh = scipy.signal.medfilt(Y, (1, 2 * lh + 1))
    Yp = scipy.signal.medfilt(Y, (2 * lp + 1, 1))
    Mh = (Yh > Yp)
    Mp = np.logical_not(Mh)
    Xh = X * Mh
    Xp = X * Mp
    xh = _invert_stft(Xh, hop_length, window)
    xp = _invert_stft(Xp, hop_length, window)
    return xh, xp

def _estimate_if(S, sr, hop_samples):
    """Estimate instantaneous frequencies from STFT matrix.
    
    This function computes instantaneous frequencies by analyzing phase
    differences between consecutive STFT frames.
    
    Args:
        S (np.ndarray): STFT matrix (complex, shape: [k_bins, n_frames])
        sr (int): Sampling rate
        hop_samples (int): Hop size of STFT analysis in samples
        
    Returns:
        np.ndarray: Instantaneous frequency matrix (shape: [k_bins, n_frames-1])
    """
    hop_sec = hop_samples / sr
    fft_size = (S.shape[0] - 1) * 2
    w_nom = np.arange(S.shape[0]) * sr / fft_size * 2 * np.pi
    w_nom = w_nom.reshape((-1, 1))
    unwrapped = np.angle(S[:, 1:]) - np.angle(S[:, 0:-1]) - w_nom * hop_sec
    wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
    w_if = w_nom + wrapped / max(hop_sec, 1e-6)
    return w_if

def _manual_stft_numpy(xh, Ha_lookup, L=2048):
    """Manual STFT implementation using NumPy.
    
    This function computes the Short-Time Fourier Transform using NumPy's
    FFT functions with custom hop size for lookup-based processing.
    
    Args:
        xh (np.ndarray): Input signal (harmonic component)
        Ha_lookup (int): Analysis hop size for lookup
        L (int): FFT size (default: 2048)
        
    Returns:
        np.ndarray: STFT matrix (complex, shape: [k_bins, n_frames])
    """
    window = scipy.signal.windows.hann(L, sym=False)
    n_frames = int(np.round((len(xh) - L) / Ha_lookup))
    k_bins = 1 + L // 2
    S_lookup = np.zeros((k_bins, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * Ha_lookup
        end = start + L
        if end > len(xh):
            break
        S_lookup[:, i] = np.fft.rfft(xh[start:end] * window)
    return S_lookup

def _on_press(key):
    '''Keyboard control for the command-line program
    
    Args:
        key (keyboard.Key): The key that was pressed, from the pynput.keyboard module
    Globals:
        alpha (float): The time-stretch factor that controls playback speed.
                      Increased with UP arrow, decreased with DOWN arrow.
    '''
    global alpha
    try:
        if key == keyboard.Key.up:
            # Increase stretch factor
            alpha += 0.01
        elif key == keyboard.Key.down:
            # Decrease stretch factor
            alpha -= 0.01
    except AttributeError:
        pass
    print(f"\rCurrent alpha: {alpha:.2f}", end="", flush=True)


class TSMRealTime:
    """Realtime hybrid audio time-stretching service.
    
    This class provides realtime audio processing capabilities using a hybrid
    approach that combines Phase Vocoder (PV) for harmonic components and
    Overlap-Add (OLA) for percussive components. It supports both baseline
    and lookup-based processing methods.
    
    Attributes:
        L_PV (int): Phase Vocoder window length (2048 samples)
        L_ola (int): OLA window length (256 samples)
        Hs_PV (int): Phase Vocoder hop size (512 samples)
        Hs_ola (int): OLA hop size (128 samples)
        sr (int): Sampling rate (22050 Hz)
        window (np.ndarray): Hann window for PV processing
        window_ola (np.ndarray): Hann window for OLA processing
        output_buffer (np.ndarray): Output audio buffer
        prev_fft (np.ndarray): Previous FFT frame for phase tracking
        prev_phase (np.ndarray): Previous phase values
        omega_nom (np.ndarray): Nominal frequencies
        den (np.ndarray): Normalization denominator
        pos (int): Current position in audio stream
        ratio (int): Ratio between PV and OLA hop sizes
    """

    # ======== Public API ========
    def __init__(self):
        """Initialize the HybridRealtime service.
        
        Sets up all necessary parameters, windows, and buffers for
        realtime audio processing.
        """
        self.L_PV =  2048
        self.L_ola = 256
        self.Hs_PV = self.L_PV // 4
        self.Hs_ola = self.L_ola // 2
        self.sr = 22050
        self.window = scipy.signal.windows.hann(self.L_PV, sym=False)
        self.window_ola = scipy.signal.windows.hann(self.L_ola, sym=False)
        self.output_buffer = np.zeros(int(self.L_PV))
        self.prev_fft = None
        self.prev_phase = np.zeros(self.L_PV // 2 + 1)
        self.omega_nom = np.arange(self.L_PV // 2 + 1) * 2 * np.pi * self.sr / self.L_PV
        self.den = _calc_sum_squared_window(self.window, self.Hs_PV)
        self.pos = 0
        self.ratio = self.Hs_PV // self.Hs_ola
    
    def update_buffer(self, pv_frame):
        """Update the output buffer with a new PV frame.
        
        Shifts the buffer and adds the new phase vocoder frame with
        proper window normalization.
        
        Args:
            pv_frame (np.ndarray): Processed PV frame to add to buffer
        """
        self.output_buffer[:-self.Hs_PV] = self.output_buffer[self.Hs_PV:]
        self.output_buffer[-self.Hs_PV:] = 0
        self.output_buffer += pv_frame
    
    def add_to_buffer(self, output_buffer, data, pos1, pos2):
        """Add data to a specific region of the output buffer.
        
        Args:
            output_buffer (np.ndarray): Target buffer to modify
            data (np.ndarray): Data to add
            pos1 (int): Start position
            pos2 (int): End position
        """
        output_buffer[pos1:pos2] += data

    def write_to_stream(self, stream, output_buffer, Hs):
        """Write audio data to the output stream.
        
        Clips the buffer to prevent overflow and converts to PCM format
        before writing to the audio stream.
        
        Args:
            stream: PyAudio output stream
            output_buffer (np.ndarray): Audio buffer to write
            Hs (int): Number of samples to write
        """
        output_buffer = np.clip(output_buffer, -1.0, 1.0)
        stream.write(_float2pcm(output_buffer[:Hs]).astype(np.int16).tobytes())
    
    def generate_lookup(self, beta, xh):
        """Generate lookup tables for phase vocoder processing.
        
        Precomputes STFT, phase, magnitude, and instantaneous frequency
        lookup tables for efficient real-time processing.
        
        Args:
            beta (float): Overlap factor for lookup analysis (e.g., 0.25)
            xh (np.ndarray): Harmonic component of input signal
            
        Returns:
            tuple: (S_phase_lookup, S_mag_lookup, w_if_lookup, Ha_lookup)
                - S_phase_lookup: Phase lookup table
                - S_mag_lookup: Magnitude lookup table  
                - w_if_lookup: Instantaneous frequency lookup table
                - Ha_lookup: Analysis hop size for lookup
        """
        Ha_lookup = int(round(beta * self.L_PV))
        S_lookup = _manual_stft_numpy(xh, Ha_lookup, L=self.L_PV)
        S_phase_lookup = np.angle(S_lookup)
        S_mag_lookup = np.abs(S_lookup)
        w_if_lookup = _estimate_if(S_lookup, self.sr, Ha_lookup)
        return S_phase_lookup, S_mag_lookup, w_if_lookup, Ha_lookup

    def phase_vocoder_full(self, xh, Ha_PV, prev_phase):
        """Perform complete phase vocoder analysis and synthesis.
        
        Processes a frame of harmonic audio using phase vocoder technique
        with phase continuity preservation.
        
        Args:
            xh (np.ndarray): Harmonic component input signal
            Ha_PV (int): Analysis hop size for phase vocoder
            prev_phase (np.ndarray): Previous phase values for continuity
            
        Returns:
            tuple: (pv_frame_mod, S) where pv_frame_mod is synthesized frame
                   and S is current FFT for next iteration
        """
        #Phase vocoder process
        pv_win = xh[self.pos:self.pos + self.L_PV] * self.window
        S = np.fft.rfft(pv_win)
        magnitude = np.abs(S)
        if self.prev_fft is not None:
            dphi = np.angle(S) - np.angle(self.prev_fft)
            dphi = dphi - self.omega_nom * (Ha_PV /self.sr)
            dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
            w_if = self.omega_nom + dphi * (self.sr / Ha_PV)
            prev_phase += w_if * (self.Hs_PV / self.sr)
        else:
            prev_phase = np.angle(S)
        X_mod = magnitude * np.exp(1j * prev_phase)
        pv_frame_mod = np.fft.irfft(X_mod) * (self.window.reshape((-1, 1)) / self.den.reshape((-1, 1))).flatten()
        return pv_frame_mod, S
    
    def phase_vocoder_lookup(self, S_phase_lookup, S_mag_lookup, w_if_lookup, Ha_PV, Ha_lookup):
        """Perform phase vocoder processing using precomputed lookup tables.
        
        Uses precomputed STFT, phase, and instantaneous frequency tables
        for efficient real-time processing with variable time-stretching.
        
        Args:
            S_phase_lookup (np.ndarray): Precomputed phase lookup table
            S_mag_lookup (np.ndarray): Precomputed magnitude lookup table
            w_if_lookup (np.ndarray): Precomputed instantaneous frequency lookup
            Ha_PV (int): Current analysis hop size
            Ha_lookup (int): Lookup analysis hop size
            
        Returns:
            np.ndarray: Synthesized phase vocoder frame
        """
        if self.prev_phase is None:
            self.prev_phase = S_phase_lookup[:, 0]
            S_mod = S_mag_lookup[:, 0] * np.exp(1j * self.prev_phase)
        else:
            frame_idx = min(int(round(self.pos / Ha_lookup)), S_mag_lookup.shape[1] - 1)
            phase_trans_idx = min(int(round((self.pos - Ha_PV) / Ha_lookup)), w_if_lookup.shape[1] - 1)
            phase_increment = w_if_lookup[:, phase_trans_idx] * (self.Hs_PV / self.sr)
            self.prev_phase += phase_increment
            S_mod = S_mag_lookup[:, frame_idx] * np.exp(1j * self.prev_phase)

        pv_frame_mod = np.fft.irfft(S_mod) * (self.window.reshape((-1, 1)) / self.den.reshape((-1, 1))).flatten()
        return pv_frame_mod

    def ola_process(self, xp, Ha_ola):
        """Process percussive component using Overlap-Add technique.
        
        Applies OLA processing to the percussive component of the signal,
        adding multiple overlapping windows to the output buffer.
        
        Args:
            xp (np.ndarray): Percussive component input signal
            Ha_ola (int): Analysis hop size for OLA processing
        """
        #OLA process
        for i in range(self.ratio):
            ola_win_synth = xp[self.pos + (Ha_ola * i):self.pos + (Ha_ola * i) + self.L_ola] * self.window_ola
            offset = i * self.Hs_ola
            self.add_to_buffer(self.output_buffer, ola_win_synth, offset, offset+self.L_ola)

    def play_hps_full(self, filename: str) -> None:
        """Play audio using hybrid baseline method with real-time controls.
        
        Combines Phase Vocoder for harmonic components and Overlap-Add for
        percussive components. Supports real-time alpha adjustment via
        keyboard controls (UP/DOWN arrows).
        
        Args:
            filename (str): Path to audio file to process
            
        Note:
            Use UP arrow to increase time-stretch factor (alpha)
            Use DOWN arrow to decrease time-stretch factor (alpha)
            Use CTRL+C to stop playback
        """
        # Load audio
        x, _ = lb.load(filename, mono=True, sr=self.sr)
        xh, xp = _harmonic_percussive_separation(x, self.sr)

        p = pyaudio.PyAudio()

        listener = keyboard.Listener(on_press=_on_press)
        listener.start()

        print("Playing audio. Press:")
        print("- 'UP' arrow to increase stretch factor")
        print("- 'DOWN' arrow to decrease stretch factor")
        print("- 'CTRL+C' to stop")

        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sr, output=True, frames_per_buffer=512)
        try:
            while self.pos <= len(xh) - self.L_PV:
                Ha_PV = int(round(self.Hs_PV / alpha))
                Ha_ola = int(round(self.Hs_ola / alpha))

                pv_frame_mod, S = self.phase_vocoder_full(xh, Ha_PV, self.prev_phase)
                self.update_buffer(pv_frame_mod)
                self.ola_process(xp, Ha_ola)
                self.write_to_stream(stream, self.output_buffer, self.Hs_PV)

                self.prev_fft = S
                self.pos += Ha_PV
        except KeyboardInterrupt:
            print("\n Stream stopped by user!")

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

    def play_hps_lookup(self, filename: str, beta: float = 0.25) -> None:
        """Play audio using hybrid lookup method with real-time controls.
        
        Uses precomputed lookup tables for efficient Phase Vocoder processing
        combined with Overlap-Add for percussive components. Supports real-time
        alpha adjustment via keyboard controls.
        
        Args:
            filename (str): Path to audio file to process
            beta (float): Overlap factor for lookup analysis (default: 0.25)
            
        Note:
            Use UP arrow to increase time-stretch factor (alpha)
            Use DOWN arrow to decrease time-stretch factor (alpha)
            Use CTRL+C to stop playback
        """
        x, _ = lb.load(filename, mono=True, sr=self.sr)
        xh, xp = _harmonic_percussive_separation(x, self.sr)
        
        S_phase_lookup, S_mag_lookup, w_if_lookup, Ha_lookup = self.generate_lookup(beta, xh) # Lookup preparation

        p = pyaudio.PyAudio()

        listener = keyboard.Listener(on_press=_on_press)
        listener.start()

        print("Playing audio. Press:")
        print("- UP arrow to increase stretch factor")
        print("- DOWN arrow to decrease stretch factor")
        print("- CTRL+C to stop")

        stream = p.open(format=pyaudio.paInt16, channels=1, rate=self.sr, output=True, frames_per_buffer=512)

        try:
            while self.pos <= len(xh) - self.L_PV:
        
                Ha_PV = int(round(self.Hs_PV / alpha))
                Ha_ola = int(round(self.Hs_ola / alpha))

                pv_frame_mod = self.phase_vocoder_lookup(S_phase_lookup, S_mag_lookup, w_if_lookup, Ha_PV, Ha_lookup)
                
                self.update_buffer(pv_frame_mod)
                self.ola_process(xp, Ha_ola)
                self.write_to_stream(stream, self.output_buffer, self.Hs_PV)

                self.pos += Ha_PV
        except KeyboardInterrupt:
            print("\n Stream stopped by user!")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


