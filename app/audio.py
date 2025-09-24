import numpy as np
import wave
import pyaudio
import threading
import librosa as lb
from scipy.signal import medfilt
import scipy
import time
import csv

import subprocess
import tempfile
import os
import subprocess
import tempfile
import os, sys

import random 

import logging
logger = logging.getLogger(__name__)

# === PyInstaller Helper ===
def get_resource_path(relative_path):
    """Get the absolute path to a resource, whether frozen or not."""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class EngineBase:
    def __init__(self, filename, gain=None, fft_size=2048, on_complete=None):
        self.filename = filename
        self.L = fft_size
        self.Hs = self.L // 4
        self.window = scipy.signal.windows.hann(self.L, sym=False)
        self.alpha = 1.0
        self.chunk_size = 512
        
        self.output_buffer = np.zeros(self.L)
        self.audio_data = None
        self.audio_sr = None
        self.stream = None
        self.p = None
        self.prev_phase = np.zeros(self.L // 2 + 1)
        self.prev_fft = None
        self.running = False
        self.thread = None
        
        self.on_complete = on_complete
        self.complete = False
        self.reset_state()

    def set_alpha(self, a):
        self.alpha = a #ax(0.1, min(a, 4.0))

    def load_audio(self):
        logging.info(f'engine loading {self.filename}')
        self.audio_data, self.audio_sr = lb.load(self.filename, sr=22050)
        self.audio_data = self.audio_data[:int(15 * self.audio_sr)]  # Limit to first 15 seconds

    def setup_audio_stream(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.audio_sr,
            output=True,
            frames_per_buffer=self.chunk_size,
        )

    def close_audio_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()

    def float2pcm(self, sig, dtype='int16'):
        sig = np.asarray(sig)
        dtype = np.dtype(dtype)
        i = np.iinfo(dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)

    def pcm2float(self, sig):
        return sig.astype(np.float32) / (2 ** 15)

    def calc_sum_squared_window(self, window, hop_length):
        assert len(window) % hop_length == 0, "Hop length must divide window length."
        numShifts = len(window) // hop_length
        den = np.zeros_like(window)
        for i in range(numShifts):
            den += np.roll(np.square(window), i * hop_length)
        return den

    def reset_state(self):
        logging.info(f'engine reset state')
        
        self.wf = wave.open(self.filename, 'rb')
        self.audio_sr = self.wf.getframerate()
        _, self.audio_sr = lb.load(self.filename, sr=22050)

        self.setup_audio_stream()
        
        self.output_buffer = np.zeros(self.L)
        self.prev_phase = np.zeros(self.L // 2 + 1)
        self.prev_fft = None
        
    def set_paused(self, paused):
        try:
            if self.stream:
                if paused and self.stream.is_active():
                    self.stream.stop_stream()
                elif not paused and not self.stream.is_active():
                    self.stream.start_stream()
        except OSError as e:
            logger.info(f"Audio stream error: {e}")
            # Optionally reinitialize the stream
            self.reinitialize_stream()
            
    def on_complete_post(self):
        self.on_complete()
    
    def start(self):
        self.reset_state()
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
            
    def _run(self):
        """Override this method in subclasses"""
        raise NotImplementedError("Subclasses must implement _run method")


class OLAEngine(EngineBase):
    def __init__(self, filename, gain=None, on_complete=None):
        super().__init__(filename, gain, fft_size=256, on_complete=on_complete)
        self.Hs = self.L // 2
        

    def _run(self):

        num_samples = self.wf.getnframes()
        pos = 0

        try:
            while self.running and pos <= num_samples - self.L:
                self.wf.setpos(pos)
                data = self.wf.readframes(self.L)
                x = np.frombuffer(data, dtype=np.int16)

                Ha = int(round(self.Hs / self.alpha))

                analysis_buffer = x * self.window
                synthesis_buffer = analysis_buffer

                self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
                self.output_buffer[-self.Hs:] = 0
                self.output_buffer[:self.L] += synthesis_buffer

                chunk_to_write = self.output_buffer[:self.Hs]
                final_chunk = np.clip(chunk_to_write, -1.0, 1.0)
                self.stream.write(self.float2pcm(final_chunk).astype(np.int16).tobytes())
                
                pos += Ha

        finally:
            self.close_audio_stream()
            self.wf.close()
            self.complete = True
            if self.on_complete:  # call callback
                self.on_complete_post()


class PVEngine(EngineBase):
    def __init__(self, filename, gain=None, on_complete=None):
        super().__init__(filename, gain=None, fft_size=2048, on_complete=on_complete)
        self.omega_nom = np.arange(self.L // 2 + 1) * 2 * np.pi * self.audio_sr / self.L
        
    def _run(self):
        num_samples = self.wf.getnframes()
        pos = int(random.random() * len(self.window))
        x, sr = lb.load(self.filename, sr=22050)
        x = x[:int(15 * self.audio_sr)]  # Limit to first 15 seconds

        
        try:  
            while self.running and pos <= num_samples - self.L:
                Ha = int(np.round(self.Hs / self.alpha))

                if len(x) < self.L:
                    x = np.pad(x, (0, self.L - len(x)))

                frame = x[pos:pos+self.L] * self.window
                S = np.fft.rfft(frame)

                # Phase Vocoder processing
                if self.prev_fft is None:
                    w_if = np.zeros_like(self.omega_nom)
                else:
                    dphi = np.angle(S) - np.angle(self.prev_fft)
                    dphi = dphi - self.omega_nom * (Ha / self.audio_sr)
                    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
                    w_if = self.omega_nom + dphi * (self.audio_sr / Ha)
                
                self.prev_phase += w_if * (self.Hs / self.audio_sr)

                X_mod = np.abs(S) * np.exp(1j * self.prev_phase)
                frame_mod = np.fft.irfft(X_mod)

                self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
                self.output_buffer[-self.Hs:] = 0
                self.output_buffer += frame_mod * self.window

                final_chunk = np.clip(self.output_buffer[:self.Hs], -1.0, 1.0)
                self.stream.write(self.float2pcm(final_chunk).astype(np.int16).tobytes())

                self.prev_fft = S
                pos += Ha

        finally:
            self.close_audio_stream()
            self.wf.close()
            self.complete = True

def invert_stft(S, hop_length, window):
    L = len(window)
    fft_size = (S.shape[0] - 1) * 2
    Sfull = np.zeros((fft_size, S.shape[1]), dtype=np.complex64)
    Sfull[0:S.shape[0], :] = S
    Sfull[S.shape[0]:, :] = np.conj(np.flipud(S[1:fft_size // 2, :]))

    frames = np.zeros_like(Sfull)
    for i in range(frames.shape[1]):
        frames[:, i] = np.fft.ifft(Sfull[:, i])
    frames = np.real(frames)

    den = calc_sum_squared_window(window, hop_length)
    frames = frames * window.reshape((-1, 1)) / den.reshape((-1, 1))

    y = np.zeros(hop_length * (frames.shape[1] - 1) + L)
    for i in range(frames.shape[1]):
        offset = i * hop_length
        y[offset:offset + L] += frames[:, i]
    return y


def calc_sum_squared_window(window, hop_length):
    numShifts = len(window) // hop_length
    den = np.zeros_like(window)
    for i in range(numShifts):
        den += np.roll(np.square(window), i * hop_length)
    return den


def harmonic_percussive_separation(x, sr=22050, fft_size=2048, hop_length=512, lh=6, lp=6):
    window = np.hanning(fft_size)
    X = lb.core.stft(x, n_fft=fft_size, hop_length=hop_length, window=window, center=False)
    Y = np.abs(X)
    Yh = medfilt(Y, (1, 2 * lh + 1))
    Yp = medfilt(Y, (2 * lp + 1, 1))
    Mh = (Yh > Yp)
    Mp = np.logical_not(Mh)
    Xh = X * Mh
    Xp = X * Mp
    xh = invert_stft(Xh, hop_length, window)
    xp = invert_stft(Xp, hop_length, window)
    return xh, xp


class HybridEngine(EngineBase):
    def __init__(self, filename, gain=None, on_complete=None):
        super().__init__(filename, gain=None, fft_size=2048, on_complete=on_complete)
        
        self.omega_nom = None
        self.den = None
        self.x = None
        self.xh = None
        self.xp = None
        self.separate_hpss()
    
    def reset_state(self):
        super().reset_state()
        self.L_ola = 256
        self.Hs_ola = self.L_ola // 2
        # self.runtimes = []
        self.setup_audio_stream()

    def separate_hpss(self):
        self.x, self.audio_sr = lb.load(self.filename, sr=22050)
        self.x = self.x[:int(15 * self.audio_sr)]  # Limit to first 15 seconds

        
        xh, xp = harmonic_percussive_separation(self.x, self.audio_sr)
        
        self.xh = xh
        self.xp = xp

        self.omega_nom = np.arange(self.L // 2 + 1) * 2 * np.pi * self.audio_sr / self.L
        self.den = self.calc_sum_squared_window(self.window, self.Hs)

    def _run(self):
        """Threading implementation for consistency with base class"""
        
        pos = 0
        ratio = self.Hs // self.Hs_ola
        windowOLA = scipy.signal.windows.hann(self.L_ola, sym=False)
        try:
            while self.running and pos <= len(self.x) - self.L:
                Ha = int(self.Hs / self.alpha)
                Ha_ola = int(self.Hs_ola / self.alpha)

                #TODO: Uncomment when done with comparing OLA parts
                # Phase Vocoder (harmonic)
                pv_win = self.x[pos:pos + self.L] * self.window
                S = np.fft.rfft(pv_win)

                if self.prev_fft is not None:
                    dphi = np.angle(S) - np.angle(self.prev_fft)
                    dphi = (dphi - self.omega_nom * (Ha / self.audio_sr) + np.pi) % (2 * np.pi) - np.pi
                    w_if = self.omega_nom + dphi * (self.audio_sr / Ha)
                    self.prev_phase += w_if * (self.Hs / self.audio_sr)
                else:
                    self.prev_phase = np.angle(S)

                X_mod = np.abs(S) * np.exp(1j * self.prev_phase)
                pv_frame_mod = np.fft.irfft(X_mod)

                self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
                self.output_buffer[-self.Hs:] = 0
                self.output_buffer += pv_frame_mod * (self.window / self.den)
                
                # OLA (percussive)
                for i in range(ratio):
                    start_i = pos + (Ha_ola * i)
                    if start_i + self.L_ola > len(self.xp):
                        continue
                    ola_win = self.xp[start_i:start_i + self.L_ola]
                    self.output_buffer[i * self.Hs_ola:i * self.Hs_ola + self.L_ola] += ola_win * windowOLA

                self.output_buffer = np.clip(self.output_buffer, -1.0, 1.0)
                
                self.stream.write(self.float2pcm(self.output_buffer[:self.Hs]).astype(np.int16).tobytes())

                self.prev_fft = S
                pos += Ha

        finally:
            self.close_audio_stream()
            self.wf.close()
            self.complete = True
            if self.on_complete:
                self.on_complete_post()
                

class OPTEngine(EngineBase):
    def __init__(self, filename, gain=None, beta=0.25, on_complete=None):
        super().__init__(filename, gain=None, fft_size=2048, on_complete=on_complete)

        self.beta = beta
        self.L_ola = 256
        self.Hs_ola = self.L_ola // 2
        self.prev_phase = np.zeros(self.L//2 + 1)
        self.PHASE_DIFF_THRESHOLD = 0.02  # radians 
        self.S_lookup = None
        self.S_phase_lookup = None
        self.S_mag_lookup = None
        self.w_if_lookup = None
        self.x = None
        self.xh = None
        self.xp = None

        self.prepare_hpss()

    def reset_state(self):
        super().reset_state()
        self.prev_phase = None
        self.setup_audio_stream()
    
    def manual_stft_numpy(self, xh, Ha_lookup, L=2048, sr=22050):
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

    def prepare_hpss(self):
        self.x, self.audio_sr = lb.load(self.filename, sr=22050)
        self.audio_data = self.x[:int(15 * self.audio_sr)]  # Limit to first 15 seconds
        # HPSS separation
        xh, xp = self.harmonic_percussive_separation(self.x, self.audio_sr)
        
        self.xh = xh
        self.xp = xp

        # Precompute STFT, phase and IF lookup for time-varying alpha
        Ha_lookup = int(round(self.beta * self.L))

        # self.S_lookup = lb.core.stft(self.xh, n_fft=self.L, hop_length=Ha_lookup, win_length=self.L, center=False, dtype=np.complex64)
        
        self.S_lookup = self.manual_stft_numpy(self.xh, Ha_lookup)
        self.S_phase_lookup = np.angle(self.S_lookup)
        self.S_mag_lookup = np.abs(self.S_lookup)
        
        self.w_if_lookup = self.estimate_instantaneous_frequency(self.S_lookup, self.audio_sr, Ha_lookup)

        self.den = self.calc_sum_squared_window(self.window, self.Hs)

    def harmonic_percussive_separation(self, x, sr, lh=6, lp=6):
        fft_size = self.L
        hop_length = self.Hs
        window = np.hanning(fft_size)

        X = lb.core.stft(x, n_fft=fft_size, hop_length=hop_length, window=window, center=False)
        Y = np.abs(X)

        Yh = medfilt(Y, (1, 2*lh+1))
        Yp = medfilt(Y, (2*lp+1, 1))

        Mh = Yh > Yp
        Mp = ~Mh

        Xh = X * Mh
        Xp = X * Mp

        xh = self.invert_stft(Xh, hop_length, window)
        xp = self.invert_stft(Xp, hop_length, window)

        return xh, xp

    def estimate_instantaneous_frequency(self, S, sr, hop_samples):
        assert sr == 22050 and S.ndim == 2, "Input must be 2D STFT at 22.05 kHz"
        hop_sec = hop_samples / sr
        fft_size = (S.shape[0] - 1) * 2
        w_nom = np.arange(S.shape[0]) * sr / fft_size * 2 * np.pi
        w_nom = w_nom.reshape((-1, 1))
        unwrapped = np.angle(S[:, 1:]) - np.angle(S[:, 0:-1]) - w_nom * hop_sec
        wrapped = (unwrapped + np.pi) % (2 * np.pi) - np.pi
        w_if = w_nom + wrapped / max(hop_sec, 1e-6)  # Prevent division by extremely small values
        return w_if

    def invert_stft(self, S, hop_length, window):
        L = len(window)
        
        # construct full stft matrix
        fft_size = (S.shape[0] - 1) * 2
        Sfull = np.zeros((fft_size, S.shape[1]), dtype=np.complex64)
        Sfull[0:S.shape[0],:] = S
        Sfull[S.shape[0]:,:] = np.conj(np.flipud(S[1:fft_size//2,:]))
        
        # compute inverse FFTs
        frames = np.zeros_like(Sfull)
        for i in range(frames.shape[1]):
            frames[:,i] = np.fft.ifft(Sfull[:,i])
        frames = np.real(frames) # remove imaginary components due to numerical roundoff
        
        # synthesis frames
        den = self.calc_sum_squared_window(window, hop_length)
        frames = frames * window.reshape((-1,1)) / den.reshape((-1,1))
        
        # reconstruction
        y = np.zeros(hop_length*(frames.shape[1]-1) + L)
        for i in range(frames.shape[1]):
            offset = i * hop_length
            y[offset:offset+L] += frames[:,i]
        return y

    def _run(self):
        pos = 0
        Ha_lookup = int(self.beta* self.L)
        ratio = self.Hs // self.Hs_ola
        windowOLA = np.hanning(self.L_ola)
        self.prev_phase = np.zeros(self.S_phase_lookup[:, 0].shape)

        try:
            while self.running and pos <= len(self.xh) - self.L:
                Ha = int(round(self.Hs/self.alpha))
                Ha_ola = int(round(self.Hs_ola/self.alpha))
                
                frame_idx = min(int(np.floor(pos / Ha_lookup)), self.S_mag_lookup.shape[1] - 1)
        
                # Get PHASE TRANSITION index (previous to current frame)
                phase_trans_idx = min(int(np.floor((pos - Ha) / Ha_lookup)), self.w_if_lookup.shape[1] - 1)
        
                phase_increment = self.w_if_lookup[:, phase_trans_idx] * (self.Hs / self.audio_sr)
                self.prev_phase += phase_increment  # Update phase correctly for current alpha
                S_mod = self.S_mag_lookup[:, frame_idx] * np.exp(1j * self.prev_phase)

                pv_frame_mod = np.fft.irfft(S_mod)

                self.output_buffer[:-self.Hs] = self.output_buffer[self.Hs:]
                self.output_buffer[-self.Hs:] = 0
                self.output_buffer += pv_frame_mod * (self.window / self.den)

                # OLA (percussive)
                for i in range(ratio):
                    ola_win_synth = self.xp[pos + (Ha_ola*i):pos +(Ha_ola*i) + self.L_ola] * windowOLA
                    offset = i *self.Hs_ola
                    self.output_buffer[offset:offset + self.L_ola] += ola_win_synth
            
                self.output_buffer = np.clip(self.output_buffer, -1.0, 1.0)  # Float32 clipping
                self.stream.write(self.float2pcm(self.output_buffer[:self.Hs]).astype(np.int16).tobytes())  # Convert to int16 at last moment
            
                pos += Ha
        finally:
            self.close_audio_stream()
            self.wf.close()
            self.complete = True
            if self.on_complete:
                self.on_complete_post()