import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# App Config
st.set_page_config(page_title="The Wobblinator", page_icon="〰️")
st.title("The Wobblinator 〰️")
st.write("Squigglevision Generator for Da Web")

# --- Core Logic ---
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

def process_single_image(image_file, fps, duration, intensity, scale):
    # Decode upload
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    fg_cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    h, w = fg_cv_image.shape[:2]
    total_frames = int(fps * duration)
    
    # Calculate wobble frequency (target ~12fps)
    updates_per_sec = 12
    frames_per_update = max(1, int(fps / updates_per_sec))
    
    map_x_base, map_y_base = np.meshgrid(np.arange(w), np.arange(h))
    map_x_base = map_x_base.astype(np.float32)
    map_y_base = map_y_base.astype(np.float32)
    
    # Video Writer Setup
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
    out = cv2.VideoWriter(tfile.name, fourcc, fps, (w, h))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tfile.name, fourcc, fps, (w, h))

    progress_bar = st.progress(0)
    current_map_x, current_map_y = None, None

    for i in range(total_frames):
        if i % frames_per_update == 0 or current_map_x is None:
            current_map_x, current_map_y = generate_noise_map(w, h, scale, intensity, map_x_base, map_y_base)
            
        frame = cv2.remap(fg_cv_image, current_map_x, current_map_y, 
                          interpolation=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_CONSTANT, 
                          borderValue=(0,0,0,0))
        
        # Composite transparent images on white
        if frame.shape[2] == 4:
            b,g,r,a = cv2.split(frame)
            overlay_color = cv2.merge((b,g,r))
            mask = a / 255.0
            bg = np.ones_like(overlay_color, dtype=np.uint8) * 255
            for c in range(3):
                bg[:,:,c] = bg[:,:,c] * (1 - mask) + overlay_color[:,:,c] * mask
            frame = bg

        out.write(frame)
        progress_bar.progress((i + 1) / total_frames)
        
    out.release()
    return tfile.name

def process_video_file(video_file, out_fps, intensity, scale):
    # Save upload to temp for OpenCV
    tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile_in.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile_in.name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(tfile_out.name, fourcc, out_fps, (w, h))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(tfile_out.name, fourcc, out_fps, (w, h))

    # Calculate wobble frequency
    updates_per_sec = 12
    frames_per_update = max(1, int(out_fps / updates_per_sec))
    
    map_x_base, map_y_base = np.meshgrid(np.arange(w), np.arange(h))
    map_x_base = map_x_base.astype(np.float32)
    map_y_base = map_y_base.astype(np.float32)
    
    current_map_x, current_map_y = None, None
    frame_count = 0
    progress_bar = st.progress(0)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_count % frames_per_update == 0 or current_map_x is None:
             current_map_x, current_map_y = generate_noise_map(w, h, scale, intensity, map_x_base, map_y_base)
             
        # Replicate borders to fill gaps
        distorted_frame = cv2.remap(frame, current_map_x, current_map_y, 
                                  interpolation=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_REPLICATE)
        out.write(distorted_frame)
        
        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))
            
    cap.release()
    out.release()
    return tfile_out.name

# --- UI Layout ---
tab1, tab2 = st.tabs(["Single Image", "Video Import"])

with tab1:
    st.header("Single Image Animation")
    uploaded_img = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    col1, col2 = st.columns(2)
    with col1:
        s_intensity = st.slider("Wobble Intensity", 1, 20, 1, key="s_int")
        s_fps = st.number_input("FPS", value=24, key="s_fps")
    with col2:
        s_scale = st.slider("Wave Scale", 5, 100, 20, key="s_sc")
        s_dur = st.number_input("Duration (sec)", value=3, key="s_dur")
        
    if uploaded_img and st.button("Generate Image Wobble"):
        with st.spinner("Processing..."):
            out_path = process_single_image(uploaded_img, s_fps, s_dur, s_intensity, s_scale)
            st.video(out_path)
            with open(out_path, 'rb') as f:
                st.download_button("Download Video", f, file_name="wobble_image.mp4")

with tab2:
    st.header("Video Animation")
    uploaded_vid = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    
    col3, col4 = st.columns(2)
    with col3:
        v_intensity = st.slider("Wobble Intensity", 1, 20, 1, key="v_int")
    with col4:
        v_scale = st.slider("Wave Scale", 5, 100, 20, key="v_sc")
        v_fps = st.number_input("Output FPS", value=24, key="v_fps")
        
    if uploaded_vid and st.button("Generate Video Wobble"):
        with st.spinner("Processing..."):
            out_path = process_video_file(uploaded_vid, v_fps, v_intensity, v_scale)
            st.video(out_path)
            with open(out_path, 'rb') as f:
                st.download_button("Download Video", f, file_name="wobble_video.mp4")