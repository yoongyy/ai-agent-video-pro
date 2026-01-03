import streamlit as st
import google.generativeai as genai
import tempfile
import os
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import re
import cv2
import numpy as np
import asyncio
import edge_tts
import json
import time
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
    TextClip,
    VideoFileClip,
    afx
)
from moviepy.video.fx import Resize

# --- 1. CONFIGURATION ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") 
WORD_LIMIT = os.getenv("MAX_WORDS_PER_IMAGE", "40") 
MAX_UPLOAD_LIMIT = int(os.getenv("MAX_UPLOAD_PHOTO", "10"))

# NEW: File Size Limits (Default to 50MB and 10MB if not set)
MAX_IMG_MB = int(os.getenv("MAX_UPLOAD_IMAGE_SIZE", "50"))
MAX_AUDIO_MB = int(os.getenv("MAX_UPLOAD_AUDIO_SIZE", "10"))

st.set_page_config(page_title="AI Video Agent Pro", layout="centered")
st.title("🇲🇾 AI Video Agent Pro")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_KEY missing in .env file!")
    st.stop()

# --- 2. LOAD VOICES (FAIL-SAFE MODE) ---
def load_voice_config():
    try:
        with open("voices.json", "r") as f:
            data = json.load(f)
            if data: return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {
        "English": {
            "Male (Christopher)": "en-US-ChristopherNeural",
            "Female (Aria)": "en-US-AriaNeural",
        },
        "Bahasa Melayu": {
            "Male (Osman)": "ms-MY-OsmanNeural",
            "Female (Yasmin)": "ms-MY-YasminNeural"
        },
        "Chinese (Mandarin)": {
            "Male (Yunxi)": "zh-CN-YunxiNeural",
            "Female (Xiaoxiao)": "zh-CN-XiaoxiaoNeural"
        }
    }

VOICE_DATA = load_voice_config()

# --- 3. SESSION STATE MANAGEMENT ---
if 'image_playlist' not in st.session_state:
    st.session_state.image_playlist = []

if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

def move_image(index, direction):
    lst = st.session_state.image_playlist
    if direction == "up" and index > 0:
        lst[index], lst[index-1] = lst[index-1], lst[index]
    elif direction == "down" and index < len(lst) - 1:
        lst[index], lst[index+1] = lst[index+1], lst[index]

def remove_image(index):
    st.session_state.image_playlist.pop(index)

# --- 4. UI INPUT SECTION ---

col_a, col_b = st.columns(2)
with col_a:
    agent_mode = st.selectbox(
        "Business Type", 
        ["Property Agent 🏠", "Car Dealer 🚗", "Retail/Product 🛍️", "General/Custom ✨"]
    )
with col_b:
    video_format = st.selectbox(
        "Video Format", 
        ["TikTok / Reels (9:16)", "YouTube / FB (16:9)"],
        index=1
    )

# --- VIDEO MODE SELECTOR ---
st.divider()
st.subheader("1. Choose Video Mode")
video_template = st.radio(
    "Select your video style:",
    ["Professional Walkthrough (Voiceover + Smooth)", "Viral Hype Edit (Music + Dynamic)"],
    index=0,
    horizontal=True
)

is_pro_mode = "Professional" in video_template

# --- AUDIO SETTINGS (CONDITIONAL) ---
st.subheader("2. Audio Settings")

if is_pro_mode:
    # PRO MODE: SHOW TTS SETTINGS
    c1, c2, c3 = st.columns(3)
    with c1:
        tts_engine = st.selectbox("TTS Engine", ("Edge-TTS (Natural)", "Google TTS (Standard)"))
    with c2:
        selected_language = st.selectbox("Language", list(VOICE_DATA.keys()))
    with c3:
        if selected_language in VOICE_DATA:
            available_personas = list(VOICE_DATA[selected_language].keys())
            selected_persona_name = st.selectbox("Voice Persona", available_personas)
            SELECTED_VOICE = VOICE_DATA[selected_language][selected_persona_name]
        else:
            SELECTED_VOICE = "en-US-ChristopherNeural"
    
    st.info("ℹ️ Pro Mode: Audio duration determines slide length.")

else:
    # VIRAL MODE: SHOW MUSIC SETTINGS
    music_option = st.radio(
        "Background Music Source",
        ["Use Default Track", "Upload MP3/WAV"],
        index=0,
        horizontal=True
    )
    
    uploaded_music = None
    default_music_path = "assets/music/upbeat.mp3" 
    
    if "Upload" in music_option:
        uploaded_music = st.file_uploader(
            f"Upload Background Music (Max {MAX_AUDIO_MB}MB)", 
            type=["mp3", "wav"]
        )
        # --- AUDIO SIZE CHECK ---
        if uploaded_music:
            if uploaded_music.size > MAX_AUDIO_MB * 1024 * 1024:
                st.error(f"❌ Audio file too big! Limit is {MAX_AUDIO_MB}MB.")
                uploaded_music = None # Reject
            else:
                st.success(f"✅ {uploaded_music.name} ready!")

    else:
        if not os.path.exists("assets/music"):
            os.makedirs("assets/music", exist_ok=True)
        
        music_files = [f for f in os.listdir("assets/music") if f.endswith('mp3')]
        if music_files:
            selected_track = st.selectbox("Select Track", music_files)
            uploaded_music = os.path.join("assets/music", selected_track)
        else:
            st.warning("⚠️ No default music found in 'assets/music'. Please upload a file.")

    st.info("ℹ️ Viral Mode: Slides are fast (2.5s) and synced to visual effects.")


if is_pro_mode:
    if "Property" in agent_mode:
        default_text = "RM 1,900/month, Seasons Garden Wangsa Maju, 3 Bedrooms."
        context_label = "Property Key Facts"
    else:
        default_text = "RM 135k, 2021 BMW 530e M-Sport, Low Mileage."
        context_label = "Car Key Facts"
    
    st.divider()
    prop_context = st.text_area(context_label, value=default_text, height=220)
else:
    prop_context = "Viral Edit" 

contact_info = st.text_input("Contact Info (End Screen)", value="Contact Kenny 012-3456789")

# --- 5. VISUALS MANAGER ---
st.divider()
st.subheader(f"3. Visuals Manager (Max {MAX_UPLOAD_LIMIT})")

if "9:16" in video_format:
    resolution_options = {"Standard (720x1280)": (720, 1280), "HD (1080x1920)": (1080, 1920)}
    default_res_idx = 0
else:
    resolution_options = {"Standard (1280x720)": (1280, 720), "HD (1920x1080)": (1920, 1080)}
    default_res_idx = 0

TARGET_W, TARGET_H = list(resolution_options.values())[0] # Default to first

uploaded_files = st.file_uploader(
    f"Add Images (Max {MAX_IMG_MB}MB each)", 
    accept_multiple_files=True, 
    type=['jpg', 'jpeg', 'png'], 
    key=f"uploader_{st.session_state.uploader_key}"
)

if uploaded_files:
    # --- IMAGE SIZE CHECK ---
    valid_files = []
    for f in uploaded_files:
        if f.size > MAX_IMG_MB * 1024 * 1024:
            st.toast(f"❌ Skipped {f.name} (Too big! Max {MAX_IMG_MB}MB)", icon="⚠️")
        else:
            valid_files.append(f)

    remaining_slots = MAX_UPLOAD_LIMIT - len(st.session_state.image_playlist)
    files_to_add = valid_files[:remaining_slots]
    
    added_count = 0
    for uploaded_file in files_to_add:
        if not any(img.name == uploaded_file.name for img in st.session_state.image_playlist):
            st.session_state.image_playlist.append(uploaded_file)
            added_count += 1
            
    if added_count > 0:
        st.session_state.uploader_key += 1
        st.rerun()

if st.session_state.image_playlist:
    st.caption(f"Total Images: {len(st.session_state.image_playlist)}")
    cols = st.columns(5)
    for i, img in enumerate(st.session_state.image_playlist):
        with cols[i % 5]:
            st.image(img, use_container_width=True)
            if st.button("❌", key=f"del_{i}"): remove_image(i); st.rerun()

# --- BACKEND FUNCTIONS ---

def remove_watermark(image_path):
    # FIXED: Replaced white box with Inpainting
    img = cv2.imread(image_path)
    if img is None: return image_path
    h, w = img.shape[:2]
    
    # 1. Create a mask for the bottom right area
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    top_y = h - 50
    left_x = w - 150
    cv2.rectangle(mask, (left_x, top_y), (w, h), (255), -1)

    # 2. Heal the image using surrounding pixels
    clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    cv2.imwrite(image_path, clean_img)
    return image_path

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
    effect_type = random.choice(['kick', 'zoom_fast', 'slide_fast'])
    
    def make_frame(get_frame, t):
        frame = get_frame(t)
        
        if effect_type == 'kick':
            progress = 1.0 - (t / 0.3) if t < 0.3 else 0
            zoom = 1.0 + (0.1 * progress)
        elif effect_type == 'zoom_fast':
            zoom = 1.0 + (0.2 * t / duration)
        else:
            zoom = 1.1 
            
        new_w = int(W / zoom)
        new_h = int(H / zoom)
        center_x, center_y = W // 2, H // 2
        
        x1 = max(0, min(center_x - new_w // 2, W - new_w))
        y1 = max(0, min(center_y - new_h // 2, H - new_h))
        
        cropped = frame[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    final_clip = clip.transform(make_frame).with_duration(duration)
    
    return final_clip

def create_end_card(text, size, duration=3):
    w, h = size
    bg_clip = ColorClip(size=(w, h), color=(0, 0, 0), duration=duration)
    font_size = 80 if w > h else 70
    
    # Simple PIL Text Card (Most reliable)
    img = PIL.Image.new("RGB", (w, h), color=(0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    # Try a few bundled fonts before falling back to the tiny default
    font = None
    for font_name in ["Arial-Bold.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"]:
        try:
            font = PIL.ImageFont.truetype(font_name, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = PIL.ImageFont.load_default()
        
    lines = textwrap.wrap(text, width=30)
    
    # Calculate total text height to center it vertically
    total_height = 0
    line_spacing = 10
    
    # Get font height
    bbox = draw.textbbox((0, 0), "Tg", font=font)
    line_h = bbox[3] - bbox[1]
    total_height = (line_h + line_spacing) * len(lines)
    
    y = (h - total_height) // 2
    
    for line in lines:
        lb = draw.textbbox((0, 0), line, font=font)
        lw = lb[2] - lb[0]
        x = (w - lw) // 2
        draw.text((x, y), line, font=font, fill="white")
        y += line_h + line_spacing
        
    return ImageClip(np.array(img)).with_duration(duration)

# --- TEXT & AI GENERATION ---
def analyze_image_with_context(image_path, context_text, role, language):
    if not is_pro_mode: return "", "" 
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    try: img = PIL.Image.open(image_path)
    except: return "Error", "View"
    
    prompt = f"Write a 1 sentence professional real estate/sales script for this image. Context: {context_text}. Language: {language}. Output: RAW TEXT ONLY."
    
    try:
        res = model.generate_content([prompt, img])
        return res.text.strip(), "Feature"
    except:
        return "Beautiful view of the property.", "View"

# --- MAIN GENERATION ---
if st.button("🚀 Generate Video"):
    if not st.session_state.image_playlist:
        st.warning("Add images first.")
    else:
        scenes = []
        status = st.empty()
        bar = st.progress(0)
        
        output_folder = "output"
        os.makedirs(output_folder, exist_ok=True)
        
        final_files = st.session_state.image_playlist
        total = len(final_files)
        
        # --- VIRAL MODE PREP ---
        bg_music_clip = None
        if not is_pro_mode:
            if isinstance(uploaded_music, str): 
                music_source = uploaded_music
            elif uploaded_music is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mus:
                    tmp_mus.write(uploaded_music.getvalue())
                    music_source = tmp_mus.name
            else:
                st.error("Please upload music for Viral Mode.")
                st.stop()
                
            try:
                bg_music_clip = AudioFileClip(music_source)
            except:
                st.error("Invalid music file.")
                st.stop()

        for i, img_file in enumerate(final_files):
            status.write(f"Processing scene {i+1}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_file.getvalue())
                img_path = tmp.name
            
            # img_path = remove_watermark(img_path)
            
            base_clip = ImageClip(img_path)
            if TARGET_W > TARGET_H:
                base_clip = base_clip.resized(width=TARGET_W)
            else:
                base_clip = base_clip.resized(height=TARGET_H)
            
            # --- BRANCHING LOGIC ---
            if is_pro_mode:
                role = "intro" if i==0 else "body"
                voice_text, label = analyze_image_with_context(img_path, prop_context, role, selected_language)
                
                tts_file = os.path.join(output_folder, f"tts_{i}.mp3")
                if "Edge" in tts_engine:
                    asyncio.run(edge_tts.Communicate(voice_text, SELECTED_VOICE).save(tts_file))
                else:
                    gTTS(voice_text, lang='en').save(tts_file) 
                
                audio = AudioFileClip(tts_file)
                duration = audio.duration + 0.5
                
                video = apply_professional_effects(base_clip, duration, i)
                video = video.with_audio(audio)
                scenes.append(video)
                
            else:
                duration = 2.5 
                video = apply_viral_effects(base_clip, duration)
                scenes.append(video)
                
            bar.progress((i+1)/total)

        if contact_info:
            scenes.append(create_end_card(contact_info, (TARGET_W, TARGET_H)))

        status.write("Stitching...")
        final_video = concatenate_videoclips(scenes, method="compose")
        
        if not is_pro_mode and bg_music_clip:
            if bg_music_clip.duration < final_video.duration:
                bg_music_clip = afx.loop(bg_music_clip, duration=final_video.duration)
            
            bg_music_clip = bg_music_clip.with_duration(final_video.duration)
            final_video = final_video.with_audio(bg_music_clip)

        filename = f"{output_folder}/video_{uuid.uuid4().hex[:8]}.mp4"
        
        final_video.write_videofile(filename, fps=24, codec="libx264", audio_codec="aac", threads=4, preset="fast")
        
        st.success("Done!")
        st.video(filename)
        with open(filename, "rb") as f:
            st.download_button("Download", f, file_name="video.mp4")
