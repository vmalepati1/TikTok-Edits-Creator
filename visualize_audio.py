import pyaudio
import numpy as np
import multiprocessing
from pydub import AudioSegment
from pydub.playback import play
from moviepy.editor import VideoFileClip
import re

duration = 12.5
audio_tiktok_link = 'https://www.tiktok.com/@glintsfx/video/7296966811646430506?q=edits&t=1704346855251'

# Download TikTok audio
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

# Get audio from the video clip
audio_array = video_clip.audio.to_soundarray()

# Convert audio to mono if it has multiple channels
if audio_array.shape[1] > 1:
    audio_array = np.mean(audio_array, axis=1)

# Convert audio to 16-bit integer
audio_array = (audio_array * (2 ** 15 - 1)).astype(np.int16)

# Define audio parameters
fs = video_clip.audio.fps
bytes_per_sample = audio_array.dtype.itemsize
channels = 1  # Mono audio

def dB(a, base=1.0):
    return 10.0 * np.log10(a / base)

def audiostream(queue, n_channels, sampling, n_bytes_per_sample):
    # open stream
    p = pyaudio.PyAudio()

    stream = p.open(
        format=p.get_format_from_width(n_bytes_per_sample),
        channels=n_channels,
        rate=sampling,
        output=True
    )

    print("Audio stream started.")

    while True:
        data = queue.get()
        if data == 'Stop':
            break
        stream.write(data)

        print("Input latency: {0}".format(stream.get_input_latency()))
        print("Output latency: {0}".format(stream.get_output_latency()))
        print("Available to read: {0}".format(stream.get_read_available()))
        print("Available to write: {0}".format(stream.get_write_available()))
    stream.close()

Q = multiprocessing.Queue()
audio_process = multiprocessing.Process(target=audiostream, args=(Q, channels, fs, bytes_per_sample))
audio_process.start()

# read data
audio_array = audio_array.reshape((-1, channels))

audio_tiktok_np = audio_array.flatten()

ch_left = 0
ch_right = 1

ch = ch_right

audio_fft = np.fft.fft(audio_tiktok_np)
freqs = np.fft.fftfreq(audio_tiktok_np.shape[0], 1.0 / fs) / 1000.0
max_freq_kHz = freqs.max()
times = np.arange(audio_tiktok_np.shape[0]) / float(fs)
fftshift = np.fft.fftshift

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure(figsize=(8.5, 11))
ax_spec_gram = fig.add_subplot(311)
ax_fft = fig.add_subplot(312)
ax_time = fig.add_subplot(313)

ax_spec_gram.specgram(audio_tiktok_np, Fs=fs, cmap='gist_heat')
ax_spec_gram.set_xlim(0, duration)
ax_spec_gram.set_ylim(0, max_freq_kHz * 1000.0)
ax_spec_gram.set_ylabel('Frequency (Hz)')

ax_fft.plot(fftshift(freqs), fftshift(dB(audio_fft)))
ax_fft.set_xlim(0, max_freq_kHz)
ax_fft.set_xlabel('Frequency (kHz)')
ax_fft.set_ylabel('dB')

ax_time.plot(times, audio_tiktok_np)
ax_time.set_xlabel('Time (s)')
ax_time.set_xlim(0, duration)
ax_time.set_ylim(-32768, 32768)

time_posn, = ax_time.plot([0, 0], [-32768, 32768], 'k')
spec_posn, = ax_spec_gram.plot([0, 0], [0, max_freq_kHz * 1000.0], 'k')


class AudioSubsetter:
    def __init__(self, audio_array, audio_device_queue, n_channels, sampling_rate, n_bytes_per_sample, chunk_dt=0.1):
        self.last_chunk = -1
        self.queue = audio_device_queue
        self.audio_dat = audio_array.tobytes()
        self.to_t = 1.0 / (sampling_rate * n_channels * n_bytes_per_sample)
        chunk = int(chunk_dt * fs) * channels * bytes_per_sample
        self.chunk0 = np.arange(0, len(self.audio_dat), chunk, dtype=int)
        self.chunk1 = self.chunk0 + chunk

    def update(self, *args):
        self.last_chunk += 1
        if self.last_chunk >= len(self.chunk0):
            self.last_chunk = 0

        i = self.last_chunk
        i0, i1 = self.chunk0[i], self.chunk1[i]
        self.queue.put(self.audio_dat[i0:i1])
        t0, t1 = i0 * self.to_t, i1 * self.to_t
        print(t0, t1)
        for line_artist in args:
            line_artist.set_xdata([t1, t1])
        args[0].figure.canvas.draw()

print("Setting up audio process")
dt = .5
playhead = AudioSubsetter(audio_tiktok_np, Q, channels, fs, bytes_per_sample, chunk_dt=dt)
timer = fig.canvas.new_timer(interval=dt * 1000.0)
timer.add_callback(playhead.update, spec_posn, time_posn)
timer.start()

plt.show()
