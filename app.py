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
from gtts import gTTS
from dotenv import load_dotenv
from moviepy import (
    ImageClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeVideoClip,
    ColorClip,
)
from moviepy.video.fx import Resize

# --- 1. CONFIGURATION ---
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") 
WORD_LIMIT = os.getenv("MAX_WORDS_PER_IMAGE", "40") 
MAX_UPLOAD_LIMIT = int(os.getenv("MAX_UPLOAD_PHOTO", "10"))

st.set_page_config(page_title="AI Video Agent Pro", layout="centered")
st.title("🇲🇾 AI Video Agent Pro")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_KEY missing in .env file!")
    st.stop()

# --- 2. LOAD VOICES FROM JSON ---
def load_voice_config():
    try:
        with open("voices.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

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

if "9:16" in video_format:
    ASPECT_NAME = "Portrait"
    resolution_options = {
        "Compact (540x960)": (540, 960),
        "Standard (720x1280)": (720, 1280),
        "HD (1080x1920)": (1080, 1920)
    }
    default_res_idx = 1 
else:
    ASPECT_NAME = "Landscape"
    resolution_options = {
        "Standard (1280x720)": (1280, 720),
        "HD (1920x1080)": (1920, 1080)
    }
    default_res_idx = 0

selected_resolution = st.selectbox("Resolution", list(resolution_options.keys()), index=default_res_idx)
TARGET_W, TARGET_H = resolution_options[selected_resolution]

st.subheader("1. Details")
if "Property" in agent_mode:
    default_text = "RM 1,900/month, Seasons Garden Wangsa Maju, 3 Bedrooms."
    context_label = "Property Key Facts"
elif "Car" in agent_mode:
    default_text = "RM 135k, 2021 BMW 530e M-Sport, Low Mileage."
    context_label = "Car Key Facts"
elif "Retail" in agent_mode:
    default_text = "New Arrival: Premium Leather Handbag. RM 250. Limited Stock."
    context_label = "Product Details"
else:
    default_text = "A beautiful sunset view at Kundasang, Sabah."
    context_label = "Context / Description"

prop_context = st.text_area(context_label, value=default_text, height=80)
contact_info = st.text_input("Contact Info (End Screen)", value="Contact Nica 013-452 9116")

# --- 5. VISUALS MANAGER ---
st.subheader(f"2. Visuals Manager (Max {MAX_UPLOAD_LIMIT})")

current_count = len(st.session_state.image_playlist)
remaining_slots = MAX_UPLOAD_LIMIT - current_count

if remaining_slots > 0:
    uploaded_files = st.file_uploader(
        f"Add Images (Can add {remaining_slots} more)", 
        accept_multiple_files=True, 
        type=['jpg', 'jpeg', 'png'], 
        key=f"uploader_{st.session_state.uploader_key}" 
    )

    if uploaded_files:
        files_to_add = uploaded_files[:remaining_slots]
        for uploaded_file in files_to_add:
            if not any(img.name == uploaded_file.name for img in st.session_state.image_playlist):
                st.session_state.image_playlist.append(uploaded_file)
        
        if len(uploaded_files) > remaining_slots:
            st.toast(f"⚠️ Limit reached! Only added {remaining_slots} images.", icon="🛑")
        
        st.session_state.uploader_key += 1
        st.rerun()
else:
    st.warning(f"🚫 Maximum limit of {MAX_UPLOAD_LIMIT} images reached. Delete some to add more.")

if st.session_state.image_playlist:
    st.caption(f"Total Images: {len(st.session_state.image_playlist)}/{MAX_UPLOAD_LIMIT}")
    
    for i, img_file in enumerate(st.session_state.image_playlist):
        with st.container():
            c1, c2, c3, c4, c5 = st.columns([1, 4, 1, 1, 1])
            with c1: st.image(img_file, width=80)
            with c2:
                st.text(f"Scene {i+1}")
                st.caption(img_file.name[:20] + "..." if len(img_file.name)>20 else img_file.name)
            with c3:
                if i > 0:
                    if st.button("⬆️", key=f"up_{i}"): move_image(i, "up"); st.rerun()
            with c4:
                if i < len(st.session_state.image_playlist) - 1:
                    if st.button("⬇️", key=f"down_{i}"): move_image(i, "down"); st.rerun()
            with c5:
                if st.button("❌", key=f"del_{i}", type="primary"): remove_image(i); st.rerun()
            st.divider()
else:
    st.info("👆 Upload images to start.")

st.subheader("3. Voice Settings")

tts_engine = st.selectbox(
    "TTS Engine",
    ("Edge-TTS (Natural)", "Google TTS (Standard)"),
    help="Edge-TTS is higher quality but may be unstable. Google TTS is a reliable backup."
)

col1, col2 = st.columns(2)
with col1:
    if VOICE_DATA:
        selected_language = st.selectbox("Language", list(VOICE_DATA.keys()))
    else:
        selected_language = "English" 
        
with col2:
    if VOICE_DATA and selected_language in VOICE_DATA:
        available_personas = list(VOICE_DATA[selected_language].keys())
        selected_persona_name = st.selectbox("Voice Persona", available_personas)
        SELECTED_VOICE = VOICE_DATA[selected_language][selected_persona_name]
    else:
        SELECTED_VOICE = "en-US-ChristopherNeural"

# --- 6. BACKEND FUNCTIONS ---

def remove_watermark(image_path):
    img = cv2.imread(image_path)
    if img is None: return image_path
    h, w = img.shape[:2]
    top_y, bottom_y = h - 60, h
    left_x, right_x = w - 200, w
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (left_x, top_y), (right_x, bottom_y), (255), -1)
    clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    cv2.imwrite(image_path, clean_img)
    return image_path

def clean_text(text):
    text = re.sub(r'^(Script|Voiceover|Line 1|Audio):', '', text, flags=re.IGNORECASE).strip()
    return text

def apply_zoom_in(clip, duration):
    return (
        clip.with_effects([Resize(lambda t: 1 + 0.04 * t)])
        .with_position(("center", "center"))
        .with_duration(duration)
    )

def apply_zoom_out(clip, duration):
    return (
        clip.with_effects([Resize(lambda t: 1.15 - 0.04 * t)])
        .with_position(("center", "center"))
        .with_duration(duration)
    )

def apply_slide(clip, duration, direction):
    if TARGET_W > TARGET_H: 
        clip = clip.resized(width=int(TARGET_W * 1.2))
        if clip.h < TARGET_H: clip = clip.resized(height=TARGET_H)
    else: 
        clip = clip.resized(height=int(TARGET_H * 1.2))
        if clip.w < TARGET_W: clip = clip.resized(width=TARGET_W)

    move_dist = 50
    if direction == 'left': pos = lambda t: (int(0 - move_dist * t), 'center')
    elif direction == 'right': 
        start_x = -(clip.w - TARGET_W) 
        pos = lambda t: (int(start_x + move_dist * t), 'center')
    elif direction == 'up': pos = lambda t: ('center', int(0 - move_dist * t))
    elif direction == 'down': 
        start_y = -(clip.h - TARGET_H)
        pos = lambda t: ('center', int(start_y + move_dist * t))
        
    return clip.with_duration(duration).with_position(pos)

# --- 7. NEW TEXT GENERATOR (UNIVERSAL & BEST FIT) ---

def get_valid_font_path():
    candidates = [
        "arial.ttf", "Arial.ttf",
        "/Library/Fonts/Arial.ttf", 
        "/System/Library/Fonts/Supplemental/Arial.ttf", 
        "C:/Windows/Fonts/arial.ttf", 
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
        "sf-pro-display-regular.otf"
    ]
    for path in candidates:
        try:
            PIL.ImageFont.truetype(path, 20)
            return path
        except OSError:
            continue
    return None

def create_text_image(text, width, height, is_label=False):
    if is_label:
        label_h = int(height * 0.15)
        img = PIL.Image.new('RGBA', (width, label_h), (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(img)
        fontsize = int(height * 0.06)
        text_color = (255, 255, 255, 255)
        stroke_color = (0, 0, 0, 255)
        max_width_pct = 0.95 
    else:
        img = PIL.Image.new('RGB', (width, height), color=(0, 0, 0))
        draw = PIL.ImageDraw.Draw(img)
        fontsize = int(height * 0.25)
        text_color = (255, 255, 255)
        stroke_color = None
        max_width_pct = 0.90 

    font_path = get_valid_font_path()

    max_safe_width = int(width * max_width_pct)
    min_fontsize = 20
    wrap_chars = 30 if not is_label else 50
    
    while fontsize > min_fontsize:
        if font_path:
            font = PIL.ImageFont.truetype(font_path, fontsize)
        else:
            font = PIL.ImageFont.load_default()
            break 

        wrapper = textwrap.TextWrapper(width=wrap_chars)
        lines = wrapper.wrap(text)
        
        max_line_w = 0
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            if line_w > max_line_w: max_line_w = line_w
        
        if max_line_w <= max_safe_width:
            break 
        
        fontsize -= 2 

    wrapper = textwrap.TextWrapper(width=wrap_chars)
    lines = wrapper.wrap(text)
    
    line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
    total_text_height = sum(line_heights) + (len(lines) * 15)
    
    current_y = (img.height - total_text_height) // 2
    
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (width - text_width) // 2
        
        if stroke_color:
            for offset in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
                draw.text((text_x + offset[0], current_y + offset[1]), line, font=font, fill=stroke_color)

        draw.text((text_x, current_y), line, font=font, fill=text_color)
        current_y += line_heights[0] + 15
        
    return np.array(img)

def create_end_card(text):
    duration = 3
    img_array = create_text_image(text, TARGET_W, TARGET_H, is_label=False)
    clip = ImageClip(img_array).with_duration(duration)
    return clip

# --- UPDATED PROMPT ---
def analyze_image_with_context(image_path, context_text, role, language, mode):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)
    try: img = PIL.Image.open(image_path)
    except: return "Error opening image", "Error"
    
    if language == "Bahasa Melayu": 
        lang_instr = (
            "Write in professional, natural Bahasa Melayu (Standard/Baku) suitable for a sales video. "
            "STRICT RULES: \n"
            "1. NO SLANG: Do NOT use words like 'Wah', 'Nah', 'Fuh', 'Amacam', 'Giler'. \n"
            "2. NO FILLERS: Do not start sentences with 'Haa', 'Okay guys'. \n"
            "3. PROFESSIONAL TONE: Speak like a high-end property agent or car showroom consultant. \n"
            "4. NUMBERS: Write 'RM 1,900' as 'Seribu Sembilan Ratus Ringgit'. Write '530e' as 'Lima Tiga Kosong E'."
        )
    elif language == "Chinese (Mandarin)":
        lang_instr = "Write in natural Simplified Chinese (Mandarin). STRICTLY NO ENGLISH."
    else: 
        lang_instr = "Write in professional English. Avoid starting every sentence with 'Check this out' or 'Look at this'."
        
    limit_instr = f"Keep it under {WORD_LIMIT} words."
    
    if "Car" in mode: persona = "Professional car sales executive."
    elif "Retail" in mode: persona = "Professional product presenter."
    elif "General" in mode: persona = "Documentary narrator."
    else: persona = "Professional property consultant."

    if role == "intro": task = f"Intro. Hook: {context_text}. {limit_instr}"
    elif role == "outro": task = f"Outro. CTA: {context_text}. {limit_instr}"
    else: task = f"Describe visual. {limit_instr}"

    prompt = f"""
    {persona} {lang_instr} Task: {task}
    Output RAW TEXT ONLY. No prefixes.
    Line 1: Spoken text
    Line 2: Label (2-4 words)
    """
    try:
        result = model.generate_content([prompt, img])
        lines = result.text.strip().split('\n')
        voice = clean_text(lines[0].strip())
        label = lines[1].strip() if len(lines) > 1 else "View"
        return voice, label
    except: return "Check this out.", "View"

async def generate_audio_edge(text, voice, retries=3):
    for attempt in range(retries):
        try:
            communicate = edge_tts.Communicate(text, voice)
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            await communicate.save(temp_audio.name)
            return temp_audio.name
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 * (attempt + 1)
                await asyncio.sleep(wait_time)
            else:
                raise e

def generate_audio_google(text, lang_name):
    lang_code = "ms" if lang_name == "Bahasa Melayu" else "zh-CN" if lang_name == "Chinese (Mandarin)" else "en"
    tts = gTTS(text=text, lang=lang_code)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

def get_audio_file(text, voice_id, lang_name, tts_engine="Edge-TTS (Natural)"):
    if "Edge-TTS" in tts_engine:
        try:
            return asyncio.run(generate_audio_edge(text, voice_id))
        except Exception as e:
            st.toast(f"⚠️ Edge-TTS failed. Falling back to Google TTS.", icon="⚠️")
            return generate_audio_google(text, lang_name)
    else:
        return generate_audio_google(text, lang_name)

def create_scene(image_file, voiceover, label, voice_id, index, language, tts_engine):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as t_img:
        t_img.write(image_file.getvalue())
        img_path = t_img.name
        
    img_path = remove_watermark(img_path)
    audio_path = get_audio_file(voiceover, voice_id, language, tts_engine)
    
    if not audio_path: 
        st.error(f"❌ Audio failed for Scene {index+1}. Skipping.")
        return None
        
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration + 0.5
    
    base_clip = ImageClip(img_path)
    if TARGET_W > TARGET_H:
        base_clip = base_clip.resized(width=TARGET_W)
        if base_clip.h < TARGET_H: base_clip = base_clip.resized(height=TARGET_H)
    else:
        base_clip = base_clip.resized(height=TARGET_H)
        if base_clip.w < TARGET_W: base_clip = base_clip.resized(width=TARGET_W)
    
    effect_types = ['zoom_in', 'slide_left', 'zoom_out', 'slide_right', 'slide_up', 'slide_down']
    chosen_effect = effect_types[index % len(effect_types)]
    
    if 'zoom' in chosen_effect:
        if chosen_effect == 'zoom_in': video_clip = apply_zoom_in(base_clip, duration)
        else: video_clip = apply_zoom_out(base_clip, duration)
    else:
        video_clip = apply_slide(base_clip, duration, chosen_effect.split('_')[1])
        
    video_clip = video_clip.with_audio(audio_clip)
    
    if language == "Chinese (Mandarin)":
        final_scene = CompositeVideoClip([video_clip], size=(TARGET_W, TARGET_H))
    else:
        # Use simpler PIL label (optional) or MoviePy textclip
        font_s = 60 if TARGET_W > TARGET_H else 50
        try:
            # Use strict PIL generation for labels too if needed
            label_img = create_text_image(label, TARGET_W, TARGET_H, is_label=True)
            txt_clip = ImageClip(label_img).with_position(("center", 0.85), relative=True).with_duration(duration)
            final_scene = CompositeVideoClip([video_clip, txt_clip], size=(TARGET_W, TARGET_H))
        except:
            final_scene = CompositeVideoClip([video_clip], size=(TARGET_W, TARGET_H))

    return final_scene

# --- 8. EXECUTION ---
if st.button("🚀 Generate Video"):
    if not st.session_state.image_playlist:
        st.warning("Please add at least one image to the Visuals Manager.")
    else:
        st.toast(f"⚙️ {ASPECT_NAME} | {SELECTED_VOICE}")
        scenes = []
        status = st.empty()
        bar = st.progress(0)
        
        final_files = st.session_state.image_playlist
        total_files = len(final_files)

        for i, img_file in enumerate(final_files):
            status.write(f"🔄 Processing Scene {i+1}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(img_file.getvalue())
                tmp_path = tmp.name
                
            role = "intro" if i == 0 else "outro" if i == total_files - 1 else "body"
            voice, label = analyze_image_with_context(tmp_path, prop_context, role, selected_language, agent_mode)
            
            clip = create_scene(img_file, voice, label, SELECTED_VOICE, i, selected_language, tts_engine)
            if clip: scenes.append(clip)
            bar.progress((i+1)/total_files)
            
        if contact_info:
            end_clip = create_end_card(contact_info)
            scenes.append(end_clip)

        if scenes:
            status.write("🎬 Stitching Final Video...")
            
            # --- FORCE OUTPUT FOLDER ---
            output_folder = os.path.abspath("output") # Use absolute path
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            
            unique_id = str(uuid.uuid4())[:8]
            output_filename = os.path.join(output_folder, f"video_{unique_id}.mp4")
            
            # Debugging: Print to terminal to be 100% sure
            print(f"Saving video to: {output_filename}")

            final = concatenate_videoclips(scenes, method="chain") 
            final.write_videofile(
                output_filename, 
                fps=24, 
                codec="libx264", 
                audio_codec="aac",
                threads=4,      
                preset='fast'   
            )
            st.success(f"✅ Video Ready!")
            
            # --- DOWNLOAD BUTTON ---
            with open(output_filename, "rb") as file:
                st.download_button(
                    label="⬇️ Download Video",
                    data=file,
                    file_name=f"video_{unique_id}.mp4",
                    mime="video/mp4"
                )
                
            st.video(output_filename)