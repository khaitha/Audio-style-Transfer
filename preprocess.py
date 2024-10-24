import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import torch

def audio_to_spectrogram(audio_path, sr=22050, n_fft=2048, hop_length=512, duration=10):
    # Load the audio file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)

    # Compute the Short-Time Fourier Transform (STFT)
    spectrogram = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Convert to magnitude spectrogram (absolute value)
    magnitude = np.abs(spectrogram)

    # Convert to log scale for better visualization
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)

    # Return the spectrogram as a PyTorch tensor
    return torch.tensor(log_magnitude, dtype=torch.float32).to(device), y, sr  # Return magnitude too


def plot_spectrogram(log_magnitude, sr, hop_length):
    # Create a matplotlib plot for the spectrogram
    plt.figure(figsize=(10, 6))

    # Display the log-magnitude spectrogram
    librosa.display.specshow(log_magnitude, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')

    # Add color bar and labels
    plt.colorbar(format="%+2.0f dB")
    plt.title("Log-Scaled Spectrogram")
    plt.tight_layout()
    plt.show()

def spectrogram_to_audio(y):
    audio = np.abs(librosa.stft(y))
    y_inv = librosa.griffinlim(audio)
    return audio

def save_audio_as_wav(audio, sr=22050, filename='output.wav'):
    sf.write(filename, audio, sr)
    print("Audio Saved")