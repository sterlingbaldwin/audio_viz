import sys
import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pydub import AudioSegment

# from scipy.fftpack import fft
# from scipy.signal import stft
from math import floor

def ave(data, n):
    return sum(data)/n

# ----- ----- ----- -----    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt32  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 1
cutoff = 10


FRAMES = 10
FPS = 1.0/FRAMES

VOL_THRESH = -30
EDGE_THRESHOLD = 5

volume_data = np.array([-30 for _ in range(FRAMES*5)])#np.zeroes((FRAMES * 5, 1))
volume_idx = 0

p = pyaudio.PyAudio()  # Create an interface to PortAudio
stream = p.open(format=sample_format,
                channels=1,
                rate=fs,
                input=True)
    
fig, axs = plt.subplots(3)

# previus_vol = -100
# threshold_v = -35

while True:
    note = False
    
    data = stream.read(chunk, exception_on_overflow = False)
    data = np.frombuffer(data, np.int32)


    
    segment = AudioSegment(
        data.tobytes(), 
        frame_rate=fs,
        sample_width=data.dtype.itemsize, 
        channels=1)
    # segment.high_pass_filter(80)
    volume = segment.dBFS

    if volume_idx < volume_data.shape[0]-1:
        volume_data[volume_idx] = volume
        volume_idx += 1
    else:
        for idx in range(volume_data.shape[0]-1):
            volume_data[idx] = volume_data[idx+1]
        volume_data[volume_idx] = volume
    
    if volume > VOL_THRESH and volume - volume_data[volume_idx-1] > EDGE_THRESHOLD:
        note = True

    axs[0].plot(volume_data) 

    # audio = data
    # data /= np.max(np.abs(data) ,axis=0)
    # data = (data - np.min(data))/np.ptp(data)
    
    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(w))

    idx = np.argmax(np.abs(w))
    freq = freqs[idx]
    freq_in_hertz = abs(freq * fs)

    l = len(w)//2
    freqs_data = abs(w[:l - floor(l/1.2)])
    axs[1].plot(freqs_data, 'r') 
    
    
    axs[0].set_title(f"freq = {freq_in_hertz}, note = {note}")


    if note:
        axs[2].cla()
        axs[2].plot(freqs_data, 'r')
        peak_indicies, props = signal.find_peaks(freqs_data, height=0.015)
        
        peak_avg = 0

        for i, peak in enumerate(peak_indicies):
            freq = freqs_data[peak]
            freq_in_hertz = abs(freq * fs)
            peak_avg += freq_in_hertz
            magnitude = props["peak_heights"][i]
            # print(f"{freq_in_hertz}hz with magnitude {magnitude:.3f}")
            print(f"{freq}hz with magnitude {magnitude:.3f}")
        
        peak_avg = peak_avg/len(peak_indicies)
        print(f"avg peak at {peak_avg}")
        print("\n\n")

    plt.pause(FPS)
    
    axs[0].cla()
    axs[1].cla()




plt.show()