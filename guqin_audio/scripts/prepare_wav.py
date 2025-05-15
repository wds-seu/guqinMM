import os
import ffmpeg
import subprocess
def convert_mp3_to_wav(mp3_folder, wav_folder, target_sr=24000):
    os.makedirs(wav_folder, exist_ok=True)
    for file in os.listdir(mp3_folder):
        if file.endswith('.mp3'):
            input_path = os.path.join(mp3_folder, file)
            output_path = os.path.join(wav_folder, file.replace('.mp3', '.wav'))
            ffmpeg.input(input_path).output(output_path, ar=target_sr, ac=1).run(overwrite_output=True)
            print(f"Converted {file} -> {output_path}")

def os_convert_mp3_to_wav(mp3_folder, wav_folder, target_sr=16000):
    os.makedirs(wav_folder, exist_ok=True)
    for file in os.listdir(mp3_folder):
        if file.endswith('.mp3'):
            input_path = os.path.join(mp3_folder, file)
            output_path = os.path.join(wav_folder, file.replace('.mp3', '.wav'))
            cmd = f"ffmpeg -i '{input_path}' -ar {target_sr} -ac 1 '{output_path}' -y"
            subprocess.call(cmd, shell=True)
            print(f"Converted {file} -> {output_path}")

if __name__ == "__main__":
    os_convert_mp3_to_wav('./data/raw', './data/processed_wav')


