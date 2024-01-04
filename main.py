import numpy as np
from scipy.io import wavfile
from scipy import signal
from pydub import AudioSegment
import pyktok as pyk
from moviepy.editor import VideoFileClip
import re
import librosa

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


