import streamlit as st
import google.generativeai as genai
import tempfile
import os
import PIL
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import numpy as np
import asyncio
import edge_tts
import json
import textwrap
import uuid
import random
from gtts import gTTS
from dotenv import load_dotenv
from moviepy import (
    ImageClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeVideoClip,
    ColorClip,
    afx
)
# Note: Ensure you are using a moviepy version compatible with these imports
from moviepy.video.fx import Resize
from typing import Tuple

# --- 1. CONFIGURATION ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_UPLOAD_LIMIT = int(os.getenv("MAX_UPLOAD_PHOTO", "10"))
MAX_IMG_MB = int(os.getenv("MAX_UPLOAD_IMAGE_SIZE", "50"))
MAX_AUDIO_MB = int(os.getenv("MAX_UPLOAD_AUDIO_SIZE", "10"))

st.set_page_config(page_title="AI Video Agent Pro", layout="centered")
st.title("🇲🇾 AI Video Agent Pro")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_KEY missing in .env file!")
    st.stop()

# --- 2. LOAD VOICES ---
def load_voice_config():
    try:
        with open("voices.json", "r") as f:
            data = json.load(f)
            if data: return data
    except: pass
    return {
        "English": {"Male (Christopher)": "en-US-ChristopherNeural", "Female (Aria)": "en-US-AriaNeural"},
        "Bahasa Melayu": {"Male (Osman)": "ms-MY-OsmanNeural", "Female (Yasmin)": "ms-MY-YasminNeural"},
        "Chinese (Mandarin)": {"Male (Yunxi)": "zh-CN-YunxiNeural", "Female (Xiaoxiao)": "zh-CN-XiaoxiaoNeural"}
    }

VOICE_DATA = load_voice_config()

# --- 3. SESSION STATE ---
if 'image_playlist' not in st.session_state: st.session_state.image_playlist = []
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

def remove_image(index):
    st.session_state.image_playlist.pop(index)

# --- 4. UI INPUT SECTION ---
col_a, col_b = st.columns(2)
with col_a:
    agent_mode = st.selectbox("Business Type", ["Property Agent 🏠", "Car Dealer 🚗", "Retail/Product 🛍️", "General/Custom ✨"])
with col_b:
    video_format = st.selectbox("Video Format", ["TikTok / Reels (9:16)", "YouTube / FB (16:9)"], index=1)

st.divider()
st.subheader("1. Choose Video Mode")
video_template = st.radio(
    "Select your video style:",
    ["Professional Walkthrough (Voiceover + Smooth)", "Viral Hype Edit (Music + Text Overlay)"],
    index=0,
    horizontal=True
)

is_pro_mode = "Professional" in video_template

# --- AUDIO SETTINGS ---
st.subheader("2. Audio Settings")
if is_pro_mode:
    c1, c2, c3 = st.columns(3)
    with c1: tts_engine = st.selectbox("TTS Engine", ("Edge-TTS (Natural)", "Google TTS (Standard)"))
    with c2: selected_language = st.selectbox("Language", list(VOICE_DATA.keys()))
    with c3:
        available = list(VOICE_DATA[selected_language].keys()) if selected_language in VOICE_DATA else ["Default"]
        selected_persona = st.selectbox("Voice Persona", available)
        SELECTED_VOICE = VOICE_DATA.get(selected_language, {}).get(selected_persona, "en-US-ChristopherNeural")
    st.info("ℹ️ Pro Mode: Audio duration determines slide length.")
else:
    music_option = st.radio("Background Music Source", ["Use Default Track", "Upload MP3/WAV"], index=0, horizontal=True)
    uploaded_music = None
    if "Upload" in music_option:
        uploaded_music = st.file_uploader(f"Upload Music (Max {MAX_AUDIO_MB}MB)", type=["mp3", "wav"])
    else:
        if not os.path.exists("assets/music"): os.makedirs("assets/music", exist_ok=True)
        music_files = [f for f in os.listdir("assets/music") if f.endswith('mp3')]
        if music_files:
            track = st.selectbox("Select Track", music_files)
            uploaded_music = os.path.join("assets/music", track)
        else:
            st.warning("⚠️ No mp3 found in assets/music")
    st.info("ℹ️ Viral Mode: Fast cuts (2.5s) with AI-generated text overlays.")

# --- CONTEXT INPUT ---
st.divider()
if "Property" in agent_mode:
    def_txt = "RM 1,900/month, Seasons Garden Wangsa Maju, 3 Bedrooms."
    lbl = "Property Key Facts"
elif "Car" in agent_mode:
    def_txt = "RM 135k, 2021 BMW 530e M-Sport, Low Mileage."
    lbl = "Car Key Facts"
else:
    def_txt = "Premium quality, Limited Edition, Buy Now."
    lbl = "Product Key Facts"

prop_context = st.text_area(lbl, value=def_txt, height=150, help="These details will be used to generate the script or text overlays.")
contact_info = st.text_input("Contact Info (End Screen)", value="Contact Kenny 012-3456789")

# --- 5. VISUALS MANAGER ---
st.divider()
st.subheader(f"3. Visuals Manager (Max {MAX_UPLOAD_LIMIT})")

res_opts = {"Standard (720x1280)": (720, 1280), "HD (1080x1920)": (1080, 1920)} if "9:16" in video_format else {"Standard (1280x720)": (1280, 720), "HD (1920x1080)": (1920, 1080)}
res_label = st.selectbox("Resolution", list(res_opts.keys()))
TARGET_W, TARGET_H = res_opts.get(res_label, list(res_opts.values())[0])

uploaded_files = st.file_uploader("Add Images", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'], key=f"up_{st.session_state.uploader_key}")

if uploaded_files:
    for f in uploaded_files:
        if len(st.session_state.image_playlist) < MAX_UPLOAD_LIMIT:
            st.session_state.image_playlist.append(f)
    st.session_state.uploader_key += 1
    st.rerun()

if st.session_state.image_playlist:
    cols = st.columns(5)
    for i, img in enumerate(st.session_state.image_playlist):
        with cols[i%5]:
            st.image(img, width="stretch")
            if st.button("❌", key=f"d_{i}"): remove_image(i); st.rerun()

# --- HELPER: ROBUST FONT LOADER ---
def get_font(fontsize):
    """
    Tries to find a bold TrueType font.
    Falls back to system paths if local files aren't found.
    This prevents the 'tiny text' issue when fallback happens.
    """
    # 0) Try the Pillow-bundled DejaVu font (if present in the environment)
    pillow_font = os.path.join(os.path.dirname(PIL.__file__), "fonts", "DejaVuSans-Bold.ttf")
    if os.path.exists(pillow_font):
        try:
            return PIL.ImageFont.truetype(pillow_font, fontsize)
        except:
            pass

    # 1) Check any project fonts placed under assets/fonts
    assets_font_dir = os.path.join("assets", "fonts")
    if os.path.isdir(assets_font_dir):
        for fname in sorted(os.listdir(assets_font_dir)):
            if fname.lower().endswith(".ttf"):
                try:
                    return PIL.ImageFont.truetype(os.path.join(assets_font_dir, fname), fontsize)
                except:
                    continue

    # 2) System fonts (Windows/Mac/Linux)
    font_names = [
        "arialbd.ttf", 
        "Arial Bold.ttf", 
        "Arial Black.ttf",
        "Helvetica-Bold.ttf", 
        "DejaVuSans-Bold.ttf",
        "impact.ttf"
    ]
    
    # Common Linux/Mac/Windows font paths
    search_paths = [
        ".", 
        "/usr/share/fonts/truetype/dejavu", 
        "/usr/share/fonts/truetype/liberation",
        "/System/Library/Fonts",
        "/System/Library/Fonts/Supplemental",
        os.path.expanduser("~/Library/Fonts"),
        "C:\\Windows\\Fonts",
        "/Library/Fonts"
    ]

    for name in font_names:
        for path in search_paths:
            full_path = os.path.join(path, name)
            if os.path.exists(full_path):
                try:
                    return PIL.ImageFont.truetype(full_path, fontsize)
                except:
                    continue
    
    # If all else fails, try loading by name alone (OS might resolve it)
    try:
        return PIL.ImageFont.truetype("arial.ttf", fontsize)
    except:
        # Absolute last resort - default PIL font (Tiny)
        # We try to avoid this by using the paths above.
        print("Warning: Using default PIL font (Text will be small).")
        return PIL.ImageFont.load_default()

# --- BACKEND FUNCTIONS ---

def apply_professional_effects(clip, duration, index):
    from moviepy.video.fx import Resize as MPResize
    effects = ['zoom_in', 'zoom_out', 'slide_left', 'slide_right']
    effect = effects[index % len(effects)]
    
    if effect == 'zoom_in':
        return clip.with_effects([MPResize(lambda t: 1 + 0.04 * t)]).with_position("center").with_duration(duration)
    elif effect == 'zoom_out':
        return clip.with_effects([MPResize(lambda t: 1.15 - 0.04 * t)]).with_position("center").with_duration(duration)
    elif 'slide' in effect:
        clip = clip.resized(width=int(TARGET_W*1.2))
        move_x = 50 if effect == 'slide_left' else -50
        return clip.with_position(lambda t: (int(move_x * t), 'center')).with_duration(duration)
    return clip.with_duration(duration)

def apply_viral_effects(clip, duration):
    W, H = clip.size
    effect_type = random.choice(['zoom_fast', 'slide_fast'])
    
    def make_frame(get_frame, t):
        frame = get_frame(t)
        if effect_type == 'zoom_fast':
            zoom = 1.0 + (0.15 * t / duration)
        else:
            zoom = 1.1
        
        new_w, new_h = int(W / zoom), int(H / zoom)
        cw, ch = W//2, H//2
        x1 = max(0, min(cw - new_w // 2, W - new_w))
        y1 = max(0, min(ch - new_h // 2, H - new_h))
        
        if effect_type == 'slide_fast':
            shift = int(50 * t/duration)
            x1 = max(0, min(x1 + shift, W - new_w))

        cropped = frame[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    return clip.transform(make_frame).with_duration(duration)

def create_caption_overlay(text, size, duration):
    w, h = size
    img = PIL.Image.new("RGBA", (w, h), (0,0,0,0))
    draw = PIL.ImageDraw.Draw(img)
    
    # Start with big font (12% height) then shrink if lines overflow width
    fontsize = int(h * 0.12)
    min_fontsize = int(h * 0.07)
    font = get_font(fontsize)

    def wrap_lines_by_width(txt, ft, max_px):
        words = txt.upper().split()
        lines, line = [], ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if draw.textlength(candidate, font=ft) <= max_px:
                line = candidate
            else:
                if line:
                    lines.append(line)
                    line = word
                else:
                    # single word too long, force it and let font shrink later
                    lines.append(word)
                    line = ""
        if line: lines.append(line)
        return lines

    max_width_px = w * 0.9
    lines = wrap_lines_by_width(text, font, max_width_px)
    # Shrink font until all lines fit within max_width_px or hit min_fontsize
    while fontsize > min_fontsize and any(draw.textlength(ln, font=font) > max_width_px for ln in lines):
        fontsize = int(fontsize * 0.92)
        font = get_font(fontsize)
        lines = wrap_lines_by_width(text, font, max_width_px)
    
    line_height = int(fontsize * 1.2)
    total_text_h = len(lines) * line_height
    # Place text closer to the bottom with a small 5% padding
    start_y = h - total_text_h - int(h * 0.05)
    
    # Proportional stroke width
    stroke_width = int(fontsize * 0.05) # 5% of font size
    
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0,0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (w - text_w) // 2
        y = start_y + (i * line_height)
        
        draw.text((x, y), line, font=font, fill="black", stroke_width=stroke_width, stroke_fill="black")
        draw.text((x, y), line, font=font, fill="white", stroke_width=0)
    
    return ImageClip(np.array(img)).with_duration(duration)

def create_end_card(text, size, duration=3):
    w, h = size
    img = PIL.Image.new("RGB", (w, h), color=(0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)

    # FIXED: End card text also big (10% of height)
    fontsize = int(h * 0.10)
    font = get_font(fontsize)
    
    chars_per_line = int((w * 0.9) / (fontsize * 0.5))
    lines = textwrap.wrap(text, width=chars_per_line)
    
    line_height = int(fontsize * 1.3)
    total_height = len(lines) * line_height
    y = (h - total_height) // 2
    
    for line in lines:
        bbox = draw.textbbox((0,0), line, font=font)
        text_w = bbox[2] - bbox[0]
        x = (w - text_w) // 2
        draw.text((x, y), line, font=font, fill="white")
        y += line_height
    clip = ImageClip(np.array(img)).with_duration(duration)
    # Ensure end card is exactly target size
    if clip.size != size:
        clip = clip.resized(size)
    return clip

def analyze_image_with_context(image_path, context_text, mode="script"):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    try: img = PIL.Image.open(image_path)
    except: return "Great view"
    
    if mode == "script":
        prompt = f"Write a 1 sentence professional sales script for this image. Context: {context_text}. Output: RAW TEXT ONLY."
    else:
        # Explicit instruction for feature/spec overlay tied to the image
        prompt = (
            "Return ONE short feature/spec phrase (2-4 words) that matches what is visible in the photo. "
            "Prefer concrete features pulled from the provided context (e.g., MULTI FUNCTION STEERING, PADDLE SHIFT, 360 CAMERA). "
            "Avoid generic hype words. UPPERCASE. Output: RAW TEXT ONLY. "
            f"Context: {context_text}"
        )
    
    try:
        res = model.generate_content([prompt, img])
        text = res.text.strip()
        # Clean up any quotes or markdown
        return text.replace('"', '').replace('*', '')
    except:
        return "BEST DEAL"

# --- HELPER: ASPECT-FILL TO TARGET FRAME ---
def fit_clip_to_frame(clip, target_w: int, target_h: int):
    """Scale to cover the target frame, then center-crop to exact size."""
    w, h = clip.size
    if w == 0 or h == 0:
        return clip
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    clip = clip.resized((new_w, new_h))
    return clip.cropped(width=target_w, height=target_h, x_center=new_w / 2, y_center=new_h / 2)

# --- MAIN GENERATION ---
if st.button("🚀 Generate Video"):
    if not st.session_state.image_playlist:
        st.warning("Add images first.")
    else:
        scenes = []
        status = st.empty()
        bar = st.progress(0)
        os.makedirs("output", exist_ok=True)
        final_files = st.session_state.image_playlist
        
        bg_music_clip = None
        if not is_pro_mode:
            source = uploaded_music if isinstance(uploaded_music, str) else None
            if not source and uploaded_music:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(uploaded_music.getvalue())
                    source = tmp.name
            if source:
                bg_music_clip = AudioFileClip(source)

        for i, img_file in enumerate(final_files):
            status.write(f"Processing scene {i+1}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_file.getvalue())
                img_path = tmp.name
            
            base_clip = fit_clip_to_frame(ImageClip(img_path), TARGET_W, TARGET_H)
            
            if is_pro_mode:
                voice_text = analyze_image_with_context(img_path, prop_context, mode="script")
                tts_file = f"output/tts_{i}.mp3"
                if "Edge" in tts_engine:
                    asyncio.run(edge_tts.Communicate(voice_text, SELECTED_VOICE).save(tts_file))
                else:
                    gTTS(voice_text).save(tts_file)
                
                audio = AudioFileClip(tts_file)
                dur = audio.duration + 0.5
                video = apply_professional_effects(base_clip, dur, i).with_audio(audio)
                scenes.append(video)
            
            else:
                caption_text = analyze_image_with_context(img_path, prop_context, mode="viral")
                dur = 2.5
                bg_clip = apply_viral_effects(base_clip, dur)
                text_clip = create_caption_overlay(caption_text, (TARGET_W, TARGET_H), dur)
                video = CompositeVideoClip([bg_clip, text_clip]).with_duration(dur)
                scenes.append(video)
                
            bar.progress((i+1)/len(final_files))

        if contact_info:
            scenes.append(create_end_card(contact_info, (TARGET_W, TARGET_H)))

        status.write("Stitching & Rendering...")
        final_video = concatenate_videoclips(scenes, method="compose")
        
        if not is_pro_mode and bg_music_clip:
            if bg_music_clip.duration < final_video.duration:
                bg_music_clip = afx.loop(bg_music_clip, duration=final_video.duration)
            final_video = final_video.with_audio(bg_music_clip.with_duration(final_video.duration))

        out_file = f"output/final_{uuid.uuid4().hex[:6]}.mp4"
        final_video.write_videofile(out_file, fps=24, codec="libx264", audio_codec="aac", preset="fast")
        
        st.success("✅ Video Ready!")
        st.video(out_file)
        with open(out_file, "rb") as f:
            st.download_button("Download MP4", f, file_name="viral_video.mp4")
