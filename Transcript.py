from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os, uuid, subprocess
import whisper
from moviepy import VideoClip, ImageClip, AudioFileClip, CompositeVideoClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deep_translator import GoogleTranslator
from pykakasi import kakasi

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== CẤU HÌNH ======
FONT_JP = "static/fonts/ipag.ttf"
FONT_EN = "static/fonts/DejaVuSans.ttf"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== KHỞI TẠO PYKAKASI ======
kks = kakasi()
kks.setMode("J", "a")
kks.setMode("H", "a")
kks.setMode("K", "a")
converter = kks.getConverter()

# ====== HÀM TẠO PHỤ ĐỀ KARAKOE 3 DÒNG ======
def make_karaoke_clip(texts, duration, fonts):
    font_jp = ImageFont.truetype(fonts["jp"], 30)
    font_romaji = ImageFont.truetype(fonts["en"], 20)
    font_en = ImageFont.truetype(fonts["en"], 23)

    w, h = 800, 200
    base_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    def make_frame(t):
        img = base_img.copy()
        draw = ImageDraw.Draw(img)

        # Karaoke tiếng Nhật
        x_jp = (w - draw.textlength(texts["jp"], font=font_jp)) // 2
        y_jp = 20
        progress = int(len(texts["jp"]) * (t / duration))
        draw.text((x_jp, y_jp), texts["jp"][:progress], font=font_jp, fill=(255, 215, 0))
        draw.text((x_jp + font_jp.getlength(texts["jp"][:progress]), y_jp),
                  texts["jp"][progress:], font=font_jp, fill=(255, 255, 255))

        # Romaji
        x_romaji = (w - draw.textlength(texts["romaji"], font=font_romaji)) // 2
        y_romaji = y_jp + 45
        draw.text((x_romaji, y_romaji), texts["romaji"], font=font_romaji, fill=(200, 200, 200))

        # English
        x_en = (w - draw.textlength(texts["en"], font=font_en)) // 2
        y_en = y_romaji + 35
        draw.text((x_en, y_en), texts["en"], font=font_en, fill=(255, 255, 255))

        return np.array(img.convert("RGB"))

    return VideoClip(make_frame, duration=duration)

# ====== API: UPLOAD & XỬ LÝ ======
@app.post("/generate")
async def generate_video(audio: UploadFile = File(...), image: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    image_path = os.path.join(UPLOAD_DIR, f"{file_id}.jpg")
    audio_path = os.path.join(UPLOAD_DIR, f"{file_id}_audio.wav")

    # Lưu file upload
    with open(video_path, "wb") as f:
        f.write(await audio.read())
    with open(image_path, "wb") as f:
        f.write(await image.read())

    # Trích xuất audio từ video bằng FFmpeg
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_path
        ], check=True)
    except subprocess.CalledProcessError:
        return {"error": "Không thể trích xuất âm thanh từ video. Kiểm tra lại file .mp4."}

    # Nhận diện giọng nói
    model = whisper.load_model("small")
    result = model.transcribe(audio_path, language="ja")

    # Tạo phụ đề karaoke
    subs = []
    for seg in result["segments"]:
        start, end = seg["start"], seg["end"]
        text_jp = seg["text"]
        text_romaji = converter.do(text_jp)
        try:
            text_en = GoogleTranslator(source="ja", target="en").translate(text_jp)
        except:
            text_en = ""
        duration = end - start
        clip = make_karaoke_clip(
            {"jp": text_jp, "romaji": text_romaji, "en": text_en},
            duration,
            {"jp": FONT_JP, "en": FONT_EN}
        )
        clip = clip.set_start(start).set_position(("center", "bottom"))
        subs.append(clip)

    # Tạo video nền
    img_clip = ImageClip(image_path).set_duration(result["segments"][-1]["end"])
    audio_clip = AudioFileClip(audio_path).audio_fadein(2).audio_fadeout(2)
    final = CompositeVideoClip([img_clip] + subs).set_audio(audio_clip)

    # Xuất video
    output_path = os.path.join(OUTPUT_DIR, f"{file_id}.mp4")
    final.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac")

    return FileResponse(output_path, media_type="video/mp4", filename="karaoke_output.mp4")