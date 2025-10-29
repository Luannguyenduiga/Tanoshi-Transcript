from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, subprocess, shutil
import whisper
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deep_translator import GoogleTranslator
from pykakasi import kakasi
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FONT_JP = r"C:\Users\Admin\Downloads\Noto_Sans_JP\static\NotoSansJP-Regular.ttf"
FONT_EN = r"C:\Windows\Fonts\Arial.ttf"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
FRAMES_DIR = "frames"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

kks = kakasi()
kks.setMode("J", "a")
kks.setMode("H", "a")
kks.setMode("K", "a")
converter = kks.getConverter()

def make_karaoke_frame(words, current_time, fonts, base_texts, bg_frame_path):
    font_romaji = ImageFont.truetype(fonts["jp"], 32)
    font_jp = ImageFont.truetype(fonts["jp"], 48)
    font_en = ImageFont.truetype(fonts["en"], 30)

    w, h = 1280, 720
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    bg = Image.open(bg_frame_path).resize((w, h - 200))
    img.paste(bg, (0, 0))

    jp_words = [word["word"].strip() for word in words]
    romaji_words = [converter.do(word["word"].strip()) for word in words]
    en_words = [""] * len(words)

    spacing = 20
    total_width = 0
    word_metrics = []

    for i in range(len(jp_words)):
        jp_w = font_jp.getlength(jp_words[i])
        romaji_w = font_romaji.getlength(romaji_words[i])
        en_w = font_en.getlength("")
        max_w = max(jp_w, romaji_w, en_w)
        word_metrics.append((jp_w, romaji_w, en_w, max_w))
        total_width += max_w + spacing

    total_width -= spacing
    x_start = (w - total_width) // 2
    y_jp = h - 180
    x_cursor = x_start
    delay = 1.0

    for i in range(len(jp_words)):
        jp_w, romaji_w, en_w, max_w = word_metrics[i]
        color = (255, 255, 255)
        if current_time >= words[i]["end"] + delay:
            color = (255, 215, 0)

        x_jp = x_cursor + (max_w - jp_w) // 2
        draw.text((x_jp, y_jp), jp_words[i], font=font_jp, fill=color)

        x_romaji = x_cursor + (max_w - romaji_w) // 2
        draw.text((x_romaji, y_jp + 70), romaji_words[i], font=font_romaji, fill=(200, 200, 200))

        x_en = x_cursor + (max_w - en_w) // 2
        draw.text((x_en, y_jp + 120), "", font=font_en, fill=(255, 255, 255))

        x_cursor += max_w + spacing

    if base_texts["en"].strip():
        draw.text(
            (w // 2, h - 40),
            base_texts["en"],
            font=font_en,
            fill=(180, 255, 180),
            anchor="mm"
        )

    return np.array(img)

model = whisper.load_model("tiny", device="cuda")
print("Whisper đang chạy trên:", model.device)

@app.post("/generate")
async def generate_video(audio: UploadFile = File(...), video: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}_bg.mp4")
    audio_path = os.path.join(UPLOAD_DIR, f"{file_id}_audio.wav")
    audio_file_path = os.path.join(UPLOAD_DIR, f"{file_id}_input")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp4")

    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    with open(video_path, "wb") as f:
        f.write(await video.read())
    with open(audio_file_path, "wb") as f:
        f.write(await audio.read())

    subprocess.run([
        "ffmpeg", "-y", "-i", audio_file_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ], check=True)

    result = model.transcribe(audio_path, language="ja", word_timestamps=True)

    fps = 15
    bg_frames_dir = os.path.join(UPLOAD_DIR, f"{file_id}_bg_frames")
    os.makedirs(bg_frames_dir, exist_ok=True)

    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(bg_frames_dir, "bg_%05d.png")
    ], check=True)

    bg_frame_files = sorted(os.listdir(bg_frames_dir))
    bg_frame_count = len(bg_frame_files)
    frame_index = 0

    for seg in result["segments"]:
        start, end = seg["start"], seg["end"]
        text_jp = seg["text"].strip()
        words = seg.get("words", [{"word": text_jp, "start": start, "end": end}])

        text_romaji = converter.do(text_jp)
        try:
            text_en = GoogleTranslator(source="ja", target="en").translate(text_jp)
        except Exception:
            text_en = ""

        duration = end - start
        total_frames = max(1, int(duration * fps))

        for i in range(total_frames):
            current_time = start + i / fps
            bg_frame_path = os.path.join(bg_frames_dir, bg_frame_files[min(frame_index, bg_frame_count - 1)])
            frame = make_karaoke_frame(
                words,
                current_time,
                {"jp": FONT_JP, "en": FONT_EN},
                {"jp": text_jp, "romaji": text_romaji, "en": text_en},
                bg_frame_path
            )
            Image.fromarray(frame).save(
                os.path.join(FRAMES_DIR, f"{file_id}_{frame_index:05d}.png")
            )
            frame_index += 1

    karaoke_video = os.path.join(OUTPUT_DIR, f"{file_id}_karaoke.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(FRAMES_DIR, f"{file_id}_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", karaoke_video
    ], check=True)

    def get_audio_duration(path):
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "json", path
        ], capture_output=True, text=True)
        try:
            return float(json.loads(result.stdout)["format"]["duration"])
        except Exception as e:
            print("Lỗi lấy thời lượng audio:", e)
            return None

    audio_duration = get_audio_duration(audio_path)
    if audio_duration is None:
        raise RuntimeError("Không lấy được thời lượng audio.")

    subprocess.run([
    "ffmpeg", "-y",
    "-i", karaoke_video, "-i", audio_path,
    "-c:v", "copy", "-c:a", "aac",
    "-map", "0:v:0", "-map", "1:a:0",
    "-shortest", output_path
], check=True)

    shutil.rmtree(bg_frames_dir, ignore_errors=True)

    print("Tổng số frame:", frame_index)
    print("Thời lượng video:", frame_index / fps, "giây")

    return FileResponse(output_path, media_type="video/mp4", filename="karaoke_output.mp4")