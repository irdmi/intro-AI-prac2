import os
import subprocess

# Полный путь к ffmpeg внутри вашего окружения lab02
ffmpeg_path = r"C:\Users\irdmi\miniconda3\envs\lab02\Library\bin\ffmpeg.exe"

video_path = "input.mp4"
output_dir = "frames"

os.makedirs(output_dir, exist_ok=True)

command = [
    ffmpeg_path,
    "-i", video_path,
    "-vf", "fps=2",
    "-q:v", "2",
    os.path.join(output_dir, "frame_%04d.jpg")
]

print("Извлечение кадров...")
subprocess.run(command, check=True)
print(f"Кадры успешно сохранены в папку {output_dir}")