from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, subprocess
import whisper
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deep_translator import GoogleTranslator
from pykakasi import kakasi
import shutil

app = FastAPI()

# ====== CORS ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== CẤU HÌNH ======
FONT_JP = r"C:\Users\Admin\Downloads\Noto_Sans_JP\static\NotoSansJP-Regular.ttf"
FONT_EN = r"C:\Windows\Fonts\Arial.ttf"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
FRAMES_DIR = "frames"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# ====== PYKAKASI ======
kks = kakasi()
kks.setMode("J", "a")
kks.setMode("H", "a")
kks.setMode("K", "a")
converter = kks.getConverter()

# ====== VẼ FRAME KARAOKE ======
def make_karaoke_frame(words, current_time, fonts, base_texts, image_path):
    font_jp = ImageFont.truetype(fonts["jp"], 48)
    font_romaji = ImageFont.truetype(fonts["jp"], 36)
    font_en = ImageFont.truetype(fonts["en"], 30)

    w, h = 1280, 720
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # === VẼ ẢNH NỀN ===
    bg = Image.open(image_path).resize((w, h - 200))
    img.paste(bg, (0, 0))

    # === VẼ PHỤ ĐỀ ===
    text_jp = base_texts["jp"].strip()
    x_jp = (w - draw.textlength(text_jp, font=font_jp)) // 2
    y_jp = h - 180
    x_cursor = x_jp

    for word in words:
        word_text = word["word"].strip()
        word_width = font_jp.getlength(word_text)
        color = (255, 255, 255)

        if current_time >= word["end"]:
            color = (255, 215, 0)

        draw.text((x_cursor, y_jp), word_text, font=font_jp, fill=color)

        x_cursor += word_width + 10

    # ROMAJI
    romaji_text = base_texts["romaji"]
    x_romaji = (w - draw.textlength(romaji_text, font=font_romaji)) // 2
    draw.text((x_romaji, y_jp + 70), romaji_text, font=font_romaji, fill=(200, 200, 200))

    # ENGLISH
    en_text = base_texts["en"]
    x_en = (w - draw.textlength(en_text, font=font_en)) // 2
    draw.text((x_en, y_jp + 120), en_text, font=font_en, fill=(255, 255, 255))

    return np.array(img)

# ====== API CHÍNH ======
@app.post("/generate")
async def generate_video(audio: UploadFile = File(...), image: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    audio_path = os.path.join(UPLOAD_DIR, f"{file_id}_audio.wav")
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp4")

    # Dọn sạch frame cũ
    if os.path.exists(FRAMES_DIR):
        shutil.rmtree(FRAMES_DIR)
    os.makedirs(FRAMES_DIR, exist_ok=True)

    # Lưu file upload
    with open(image_path, "wb") as f:
        f.write(await image.read())

    audio_file_path = os.path.join(UPLOAD_DIR, f"{file_id}_input")
    with open(audio_file_path, "wb") as f:
        f.write(await audio.read())

    # === 1. Trích xuất audio nếu là video ===
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_file_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ], check=True)

    # === 2. Whisper: tách từ có timestamp ===
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, language="ja", word_timestamps=True)

    fps = 30
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
        total_frames = int(duration * fps)

        for i in range(total_frames):
            current_time = start + i / fps
            frame = make_karaoke_frame(
                words,
                current_time,
                {"jp": FONT_JP, "en": FONT_EN},
                {"jp": text_jp, "romaji": text_romaji, "en": text_en},
                image_path
            )
            Image.fromarray(frame).save(
                os.path.join(FRAMES_DIR, f"{file_id}_{frame_index:05d}.png")
            )
            frame_index += 1

    # === 3. Render frame → video karaoke ===
    karaoke_video = os.path.join(OUTPUT_DIR, f"{file_id}_karaoke.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(FRAMES_DIR, f"{file_id}_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", karaoke_video
    ], check=True)

    # === 4. Ghép audio vào video đã có ảnh nền + phụ đề ===
    subprocess.run([
        "ffmpeg", "-y",
        "-i", karaoke_video, "-i", audio_path,
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        "-pix_fmt", "yuv420p", output_path
    ], check=True)

    return FileResponse(output_path, media_type="video/mp4", filename="karaoke_output.mp4")