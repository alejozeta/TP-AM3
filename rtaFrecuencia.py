import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.io import wavfile
import sqlite3

from pydub import AudioSegment

def open_aup3(file_path):
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()
    
    # Query to extract the audio data from the 'sampleblocks' table
    cursor.execute("SELECT samples FROM sampleblocks")
    data_rows = cursor.fetchall()
    
    # Extract and concatenate all audio samples into a single NumPy array
    data = b''.join(row[0] for row in data_rows if row[0] is not None)
    
    # Convert binary data to a NumPy array with int16 format (assuming 16-bit audio)
    audio_data = np.frombuffer(data, dtype=np.int16)
    
    conn.close()
    sample_rate = 44100  # Default sample rate if not specified
    return sample_rate, audio_data

def open_mp3(file_path):
    # Load the .mp3 file
    audio = AudioSegment.from_mp3(file_path)
    
    # Convert to numpy array
    data = np.array(audio.get_array_of_samples())
    
    # Get sample rate
    sample_rate = audio.frame_rate
    
    return sample_rate, data

def open_wav(file_path):
    with open(file_path, 'rb') as file:
        sample_rate, audio_data = wavfile.read(file)
    return sample_rate, audio_data

def apply_fft(data, sample_rate):
    N = len(data)
    # Apply FFT and get the frequency domain representation
    y_f = np.fft.rfft(data)
    f = np.linspace(0, sample_rate // 2, num=len(y_f))
    return f, y_f

def compute_frequency_response(y_aup3, y_wav):
    # Print the lengths of both input arrays for debugging
    print("Length of y_aup3:", len(y_aup3))
    print("Length of y_wav:", len(y_wav))

    # Determine the maximum length of the two arrays
    max_length = max(len(y_aup3), len(y_wav))

    # Zero-pad the shorter signal to match the length of the longer one
    if len(y_aup3) < max_length:
        y_aup3 = np.pad(y_aup3, (0, max_length - len(y_aup3)), mode='constant')
    if len(y_wav) < max_length:
        y_wav = np.pad(y_wav, (0, max_length - len(y_wav)), mode='constant')
    
    # Print the new lengths after zero-padding
    print("Length after zero-padding y_aup3:", len(y_aup3))
    print("Length after zero-padding y_wav:", len(y_wav))
    
    # Avoid division by zero by adding a small epsilon value
    epsilon = 1e-10
    H_jw = y_aup3 / (y_wav + epsilon)
    return H_jw


def plotting_function(file_path_output, file_path_input):
    # sample_rate_aup3, data_aup3 = open_aup3(file_path_output)

    sample_rate_aup3, data_aup3 = open_mp3(file_path_output)
    sample_rate_wav, data_wav = open_wav(file_path_input)

    # Apply FFT to both signals
    xf_aup3, Y_aup3 = apply_fft(data_aup3, sample_rate_aup3)
    xf_wav, X_wav = apply_fft(data_wav, sample_rate_wav)

    # Compute the frequency response H(jw)
    H_jw = compute_frequency_response(Y_aup3, X_wav)
    magnitude_H = np.abs(H_jw)
    phase_H = np.angle(H_jw)

    # Truncate the frequency vector to match the length of H_jw
    min_length = len(H_jw)
    xf_aup3 = xf_aup3[:min_length]

    flag = True ## Change to False to avoid plotting

    if flag: 
        # Plot the frequency domain comparison
        fig, ax = plt.subplots()
        ax.plot(xf_aup3, np.abs(Y_aup3[:min_length]), label='Processed Signal (.aup3)')
        ax.plot(xf_wav[:min_length], np.abs(X_wav[:min_length]), label='Original Signal (.wav)', linestyle='--')
        ax.set_xlabel(r'$Frequency \;[Hz]$')
        ax.set_ylabel(r'$Amplitude$')
        ax.legend()
        plt.title("Frequency Domain Signals")
        plt.show()

        # Plot the frequency response H(jw)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(xf_aup3, magnitude_H)
        ax1.set_xlabel(r'$Frequency \;[Hz]$')
        ax1.set_ylabel(r'$|H(j\omega)|$')
        ax1.set_title("Magnitude of H(jw)")

        ax2.plot(xf_aup3, phase_H)
        ax2.set_xlabel(r'$Frequency \;[Hz]$')
        ax2.set_ylabel(r'$\angle H(j\omega)$ (radians)')
        ax2.set_title("Phase of H(jw)")

        plt.tight_layout()
        plt.show()

    return 0


# Example usage
# file_path_output = 'sine_wave.aup3'
# file_path_input = 'sine_wave.wav'

# file_path_output = 'square_wave.aup3'
# file_path_input = 'square_wave.wav'

# file_path_output = 'sawtooth_wave.aup3'
# file_path_input = 'sawtooth_wave.wav'

file_path_output = 'sawtooth_wave.mp3'
file_path_input = 'sawtooth_wave.wav'

plotting_function(file_path_output, file_path_input)
