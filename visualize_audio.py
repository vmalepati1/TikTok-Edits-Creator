import numpy as np
from scipy.io import wavfile
from scipy import signal
from pydub import AudioSegment
import pyktok as pyk
from moviepy.editor import VideoFileClip
import re
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sklearn.preprocessing
from pydub import AudioSegment
from pydub.playback import play

def normalize(x, axis = 0):
  return sklearn.preprocessing.minmax_scale(x, axis = axis)

def update_line(num, line, centroids, frames, sampling_rate):
    line.set_xdata([frames[num], frames[num]])  # Update x-coordinate based on time
    line.set_ydata(centroids[:, num])
    return line,

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

# Compute spectral centoids (average frequency at each time)
spectral_centroids = librosa.feature.spectral_centroid(y=mono_audio, sr=sampling_rate)[0]
print(f'Spectral centroids audio shape: {spectral_centroids.shape}')

plt.figure(figsize=(12, 4))
frames = range(len(spectral_centroids))

t = librosa.frames_to_time(frames, sr=sampling_rate)

# Plot spectral centroids
librosa.display.waveshow(mono_audio, sr = sampling_rate, alpha = 0.4)

normalized_centroids = normalize(spectral_centroids)

# Plot spectral centroids as a blue line
plt.plot(t, normalized_centroids, color='b')

# Initialize the red line at time 0
red_line, = plt.plot([0, 0], [0, 1], color='r', lw=2)

# Set plot properties
plt.title('Waveform with Spectral Centroids')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude / Spectral Centroids')

# Convert the numpy array to an AudioSegment
audio_array = (mono_audio * (2**15 - 1)).astype(np.int16)
audio_segment = AudioSegment(audio_array.tobytes(), frame_rate=sampling_rate, sample_width=audio_array.dtype.itemsize, channels=1)

# Play the audio
play(audio_segment)

### Create an animation to update the red line position
##ani = animation.FuncAnimation(
##    plt.gcf(),
##    update_line,
##    len(frames),
##    fargs=(red_line, normalized_centroids, frames, sampling_rate),
##    interval=sampling_rate / len(frames) * 1000,  # Interval in milliseconds
##    blit=True
##)

plt.show()
