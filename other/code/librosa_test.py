import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


y, sr = librosa.load(librosa.util.example_audio_file())

D = librosa.feature.melspectrogram(y=y, sr=sr)

'''
D = np.abs(librosa.stft(y))
'''
librosa.display.specshow(librosa.amplitude_to_db(D,
    ref=np.max),
    y_axis='log', x_axis='time')
plt.title('Mel scaled spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()

plt.show()