import numpy as np
from scipy.io import wavfile
from scipy import signal
from pydub import AudioSegment
import pyktok as pyk
from moviepy.editor import VideoFileClip
import re
import librosa
import matplotlib.pyplot as plt
import sklearn.preprocessing

def normalize(x, axis = 0):
  return sklearn.preprocessing.minmax_scale(x, axis = axis)

audio_tiktok_link = 'https://www.tiktok.com/@glintsfx/video/7296966811646430506?q=edits&t=1704346855251'
##
##pyk.specify_browser('chrome')
##pyk.save_tiktok(audio_tiktok_link, True, 'audio_video_data.csv', 'chrome')

match = re.search(r'@(\w+)/video/(\d+)', audio_tiktok_link)

mp4_filename = ''

if match:
    username = match.group(1)
    video_id = match.group(2)

    # Construct the mp4_filename
    mp4_filename = f'@{username}_video_{video_id}.mp4'
else:
    print("Invalid TikTok link.")
    exit()

video_clip = VideoFileClip(mp4_filename)

audio_array = video_clip.audio.to_soundarray()

sampling_rate = video_clip.audio.fps

print(f'Sampling Rate: {sampling_rate} Hz')

# Collapse stereo audio array into a mono audio array with 1 channel
mono_audio = np.mean(audio_array, axis=1)

print(f'Mono audio shape: {mono_audio.shape}')

X = librosa.stft(mono_audio)
X_db = librosa.amplitude_to_db(abs(X))

# Plot spectogram
plt.figure(figsize=(10, 5))
librosa.display.specshow(X_db, sr = sampling_rate, x_axis = 'time', y_axis = 'hz')
plt.colorbar(format='%+2.0f dB')

# Compute spectral centoids (average frequency at each time)
spectral_centroids = librosa.feature.spectral_centroid(y=mono_audio, sr=sampling_rate)[0]
print(f'Spectral centroids audio shape: {spectral_centroids.shape}')

plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))

t = librosa.frames_to_time(frames, sr=sampling_rate)

# Plot spectral centroids
librosa.display.waveshow(mono_audio, sr = sampling_rate, alpha = 0.4)
plt.plot(t, normalize(spectral_centroids), color = 'b')
plt.title('Waveform with Spectral Centroids')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude / Spectral Centroids')
plt.show()

# Percentile within which frequencies this low are considered beat drops
beat_drop_centroid_thresh = 5

threshold = np.percentile(spectral_centroids, beat_drop_centroid_thresh)

beat_drop_indices = np.where(spectral_centroids < threshold)[0]

beat_drop_timestamps = librosa.frames_to_time(beat_drop_indices, sr=sampling_rate)

print(beat_drop_timestamps)
