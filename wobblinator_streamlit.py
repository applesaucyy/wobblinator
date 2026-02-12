import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import subprocess
import shutil
import gc

st.set_page_config(page_title="The Wobblinator", page_icon="〰️", layout="centered")

MAX_FILE_SIZE_MB = 100
MAX_DIMENSION = 720

st.markdown("""
<style>
    :root {
        --bg-color: #0f0f13;
        --glass-bg: rgba(255, 255, 255, 0.05);
        --glass-border: rgba(255, 255, 255, 0.1);
        --text-main: #ffffff;
        --accent-color: #6366f1;
    }

    .stApp {
        background-color: var(--bg-color);
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(236, 72, 153, 0.1) 0%, transparent 40%);
        color: var(--text-main);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    .block-container {
        background: var(--glass-bg);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--glass-border);
        border-radius: 24px;
        padding: 3rem !important;
        margin-top: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36);
        max-width: 800px;
    }

    header, footer, #MainMenu {visibility: hidden;}

    h1 {
        font-weight: 300;
        letter-spacing: 1px;
        text-align: center;
        text-shadow: 0 0 20px rgba(255,255,255,0.1);
    }
    h1 span {
        font-weight: 700;
        color: var(--accent-color);
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.03);
        border-radius: 12px;
        border: 1px solid var(--glass-border);
        color: #a1a1aa;
        padding: 10px 20px;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--accent-color) !important;
        color: white !important;
        border-color: var(--accent-color);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    .stButton > button {
        background: var(--glass-bg);
        border: 1px solid var(--glass-border);
        color: white;
        border-radius: 16px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
        width: 100%;
        backdrop-filter: blur(4px);
    }
    .stButton > button:hover {
        background: rgba(255,255,255,0.1);
        border-color: var(--accent-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }

    .stSlider > div > div > div > div { background-color: var(--accent-color); }
    .stNumberInput input { color: white; background: rgba(0,0,0,0.2); border-radius: 8px; }
    
    .stProgress > div > div > div > div {
        background-image: none !important;
        background-color: var(--accent-color) !important;
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(0,0,0,0.2);
        border: 2px dashed var(--glass-border);
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(255,255,255,0.4);
        background: rgba(255,255,255,0.02);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>The <span>Wobblinator</span></h1>", unsafe_allow_html=True)

def cleanup_files(*filepaths):
    for path in filepaths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except:
                pass

def check_file_size(file):
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File too large. Please keep uploads under {MAX_FILE_SIZE_MB}MB to prevent crashes.")
        return False
    return True

def convert_to_h264(input_path):
    fd, output_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    if shutil.which("ffmpeg") is None:
        st.error("Error: FFmpeg not installed on server.")
        return input_path

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-acodec", "aac",
        "-movflags", "+faststart",
        output_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except subprocess.CalledProcessError:
        return input_path

def resize_if_needed(image, max_dim=MAX_DIMENSION):
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def generate_noise_map(w, h, wave_scale, intensity, map_x_base, map_y_base):
    safe_scale = max(5, wave_scale)
    grid_w = max(3, int(w / safe_scale))
    grid_h = max(3, int(h / safe_scale))
    
    noise_x = np.random.uniform(-1, 1, (grid_h, grid_w)).astype(np.float32)
    noise_y = np.random.uniform(-1, 1, (grid_h, grid_w)).astype(np.float32)

    noise_x_full = cv2.resize(noise_x, (w, h), interpolation=cv2.INTER_CUBIC)
    noise_y_full = cv2.resize(noise_y, (w, h), interpolation=cv2.INTER_CUBIC)

    map_x = map_x_base + (noise_x_full * intensity * 3) 
    map_y = map_y_base + (noise_y_full * intensity * 3)
    return map_x, map_y

def process_single_image(image_file, background_file, fps, duration, intensity, scale):
    tfile_raw_path = None
    final_path = None
    
    try:
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        fg_cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        
        # Resize immediately to save memory
        fg_cv_image = resize_if_needed(fg_cv_image)
        h, w = fg_cv_image.shape[:2]
        
        bg_cv_image = None
        if background_file:
            bg_bytes = np.asarray(bytearray(background_file.read()), dtype=np.uint8)
            bg_raw = cv2.imdecode(bg_bytes, cv2.IMREAD_UNCHANGED)
            bg_cv_image = cv2.resize(bg_raw, (w, h)) # Force match FG size
            if len(bg_cv_image.shape) == 3 and bg_cv_image.shape[2] == 4:
                bg_cv_image = cv2.cvtColor(bg_cv_image, cv2.COLOR_BGRA2BGR)
            elif len(bg_cv_image.shape) == 2:
                bg_cv_image = cv2.cvtColor(bg_cv_image, cv2.COLOR_GRAY2BGR)

        total_frames = int(fps * duration)
        frames_per_update = max(1, int(fps / 12))
        
        map_x_base, map_y_base = np.meshgrid(np.arange(w), np.arange(h))
        map_x_base = map_x_base.astype(np.float32)
        map_y_base = map_y_base.astype(np.float32)
        
        tfile_raw = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_raw_path = tfile_raw.name
        tfile_raw.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(tfile_raw_path, fourcc, fps, (w, h))

        progress_bar = st.progress(0)
        current_map_x, current_map_y = None, None

        for i in range(total_frames):
            if i % frames_per_update == 0 or current_map_x is None:
                current_map_x, current_map_y = generate_noise_map(w, h, scale, intensity, map_x_base, map_y_base)
                
            frame = cv2.remap(fg_cv_image, current_map_x, current_map_y, 
                              interpolation=cv2.INTER_LINEAR, 
                              borderMode=cv2.BORDER_CONSTANT, 
                              borderValue=(0,0,0,0))
            
            if frame.shape[2] == 4:
                b,g,r,a = cv2.split(frame)
                overlay = cv2.merge((b,g,r))
                mask = a / 255.0
                
                if bg_cv_image is not None:
                    base = bg_cv_image
                else:
                    base = np.ones_like(overlay, dtype=np.uint8) * 255
                
                mask_3d = np.dstack([mask]*3)
                frame = (base * (1.0 - mask_3d) + overlay * mask_3d).astype(np.uint8)

            out.write(frame)
            
            del frame
            if i % 10 == 0:
                gc.collect()
                
            progress_bar.progress((i + 1) / total_frames)
            
        out.release()
        gc.collect()
        
        final_path = convert_to_h264(tfile_raw_path)
        
        with open(final_path, 'rb') as f:
            video_bytes = f.read()
            
        return video_bytes
        
    finally:
        cleanup_files(tfile_raw_path, final_path)
        gc.collect()

def process_video_file(video_file, out_fps, intensity, scale):
    tfile_in_path = None
    tfile_raw_path = None
    final_path = None
    cap = None
    out = None
    
    try:
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in_path = tfile_in.name
        tfile_in.write(video_file.getbuffer())
        tfile_in.close()
        
        cap = cv2.VideoCapture(tfile_in_path)
        if not cap.isOpened():
            st.error("Error reading video file.")
            return None
            
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate resize target if too big
        scale_factor = 1.0
        if max(w_orig, h_orig) > MAX_DIMENSION:
            scale_factor = MAX_DIMENSION / max(w_orig, h_orig)
            w = int(w_orig * scale_factor)
            h = int(h_orig * scale_factor)
        else:
            w, h = w_orig, h_orig
        
        tfile_raw = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_raw_path = tfile_raw.name
        tfile_raw.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tfile_raw_path, fourcc, out_fps, (w, h))

        frames_per_update = max(1, int(out_fps / 12))
        
        map_x_base, map_y_base = np.meshgrid(np.arange(w), np.arange(h))
        map_x_base = map_x_base.astype(np.float32)
        map_y_base = map_y_base.astype(np.float32)
        
        current_map_x, current_map_y = None, None
        frame_count = 0
        progress_bar = st.progress(0)
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Resize frame if needed
            if scale_factor < 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            
            if frame_count % frames_per_update == 0 or current_map_x is None:
                 current_map_x, current_map_y = generate_noise_map(w, h, scale, intensity, map_x_base, map_y_base)
                 
            distorted = cv2.remap(frame, current_map_x, current_map_y, 
                                  interpolation=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_REPLICATE)
            out.write(distorted)
            
            del frame, distorted
            if frame_count % 10 == 0:
                gc.collect()
            
            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                
        out.release()
        cap.release()
        gc.collect()
        
        final_path = convert_to_h264(tfile_raw_path)
        
        with open(final_path, 'rb') as f:
            video_bytes = f.read()
            
        return video_bytes
        
    finally:
        if cap and cap.isOpened(): cap.release()
        if out and out.isOpened(): out.release()
        cleanup_files(tfile_in_path, tfile_raw_path, final_path)
        gc.collect()

tab1, tab2 = st.tabs(["Single Image", "Video Import"])

with tab1:
    st.header("Single Image Animation")
    
    use_2_layer = st.checkbox("Enable 2-Layer Mode (Static Background)", value=False)
    
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        fg_label = "Foreground (Lineart)" if use_2_layer else "Upload Image"
        uploaded_img = st.file_uploader(fg_label, type=['png', 'jpg', 'jpeg'], key="fg")
    
    with col_u2:
        uploaded_bg = None
        if use_2_layer:
            uploaded_bg = st.file_uploader("Background (Static)", type=['png', 'jpg', 'jpeg'], key="bg")
    
    col1, col2 = st.columns(2)
    with col1:
        s_intensity = st.slider("Wobblemeat Intensity", 1, 20, 1, key="s_int")
        s_fps = st.number_input("FPS", value=12, key="s_fps")
    with col2:
        s_scale = st.slider("Wave Scale", 5, 100, 20, key="s_sc")
        s_dur = st.number_input("Duration (sec)", value=3, key="s_dur")
        
    if uploaded_img and st.button("Generate Image Wobble"):
        if check_file_size(uploaded_img):
            if use_2_layer and uploaded_bg:
                if not check_file_size(uploaded_bg):
                    st.stop()
            
            if use_2_layer and not uploaded_bg:
                st.error("Please upload a background image for 2-layer mode.")
            else:
                with st.spinner("Processing..."):
                    video_bytes = process_single_image(uploaded_img, uploaded_bg, s_fps, s_dur, s_intensity, s_scale)
                    if video_bytes:
                        st.video(video_bytes)
                        st.download_button("Download Video", data=video_bytes, file_name="wobble_image.mp4", mime="video/mp4")

with tab2:
    uploaded_vid = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    
    col3, col4 = st.columns(2)
    with col3:
        v_intensity = st.slider("Wobblemeat Intensity", 1, 20, 1, key="v_int")
    with col4:
        v_scale = st.slider("Wave Scale", 5, 100, 20, key="v_sc")
        v_fps = st.number_input("Output FPS", value=24, key="v_fps")
        
    if uploaded_vid and st.button("Generate Video Wobble"):
        if check_file_size(uploaded_vid):
            with st.spinner("Processing..."):
                video_bytes = process_video_file(uploaded_vid, v_fps, v_intensity, v_scale)
                if video_bytes:
                    st.video(video_bytes)
                    st.download_button("Download Video", data=video_bytes, file_name="wobble_video.mp4", mime="video/mp4")
