import os
import soundfile
import librosa

root_path = "./audio/wavs"
audio_dir = os.listdir(root_path)

for wav in audio_dir:
    y, sr = librosa.load(path=os.path.join(root_path, wav), sr=22050)
    soundfile.write(os.path.join(root_path, wav), y, sr)
    print(f"{wav} downsampled to {22050}Hz.")