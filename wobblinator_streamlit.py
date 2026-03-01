import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import zipfile
import shutil

# App Config
st.set_page_config(page_title="The Wobblinator v4.2", layout="centered")
st.title("The Wobblinator v4.2")
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

def composite_on_white(cv_img):
    """Composites images with alpha channels onto a solid white background."""
    if len(cv_img.shape) == 3 and cv_img.shape[2] == 4:
        img_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA))
        bg_white = Image.new("RGBA", img_pil.size, (255, 255, 255, 255))
        bg_white.alpha_composite(img_pil)
        return cv2.cvtColor(np.array(bg_white), cv2.COLOR_RGBA2BGR)
    elif len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
        return cv_img
    else:
        return cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)

def export_frames(frames, fps, fmt, w, h):
    """Handles exporting the generated frames into the requested format."""
    if fmt == "MP4 Video":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(tfile.name, fourcc, fps, (w, h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tfile.name, fourcc, fps, (w, h))
            
        for frame in frames:
            out.write(frame)
        out.release()
        return tfile.name, "video/mp4", "wobbled.mp4"
        
    elif fmt == "GIF Animation":
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        pil_frames[0].save(tfile.name, save_all=True, append_images=pil_frames[1:], duration=int(1000/fps), loop=0)
        return tfile.name, "image/gif", "wobbled.gif"
        
    elif fmt == "PNG Sequence (ZIP)":
        temp_dir = tempfile.mkdtemp()
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(temp_dir, f"image_{i+1:04d}.png"), frame)
            
        zip_base = tempfile.NamedTemporaryFile(delete=False).name
        shutil.make_archive(zip_base, 'zip', temp_dir)
        shutil.rmtree(temp_dir) # Cleanup raw images
        return f"{zip_base}.zip", "application/zip", "wobbled_sequence.zip"

def generate_quick_preview(cv_img, intensity, scale):
    """Generates a 2-second 12FPS preview GIF of the current settings."""
    h, w = cv_img.shape[:2]
    
    # Scale down preview if the image is extremely large to save computation time
    max_dim = 600
    if max(h, w) > max_dim:
        ratio = max_dim / max(h, w)
        w, h = int(w * ratio), int(h * ratio)
        cv_img = cv2.resize(cv_img, (w, h))
        
    fps = 12
    total_frames = fps * 2 # 2 Seconds of wobble
    
    map_x_base, map_y_base = np.meshgrid(np.arange(w), np.arange(h))
    map_x_base, map_y_base = map_x_base.astype(np.float32), map_y_base.astype(np.float32)

    frames = []
    for _ in range(total_frames):
        map_x, map_y = generate_noise_map(w, h, scale, intensity, map_x_base, map_y_base)
        frame = cv2.remap(cv_img, map_x, map_y, 
                          interpolation=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REPLICATE)
        frame = composite_on_white(frame)
        frames.append(frame)
        
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    pil_frames[0].save(tfile.name, save_all=True, append_images=pil_frames[1:], duration=int(1000/fps), loop=0)
    return tfile.name

def get_first_frame(source_type, source_data):
    """Extracts the first frame from a video, GIF, or sequence to use as a preview thumbnail."""
    frame = None
    if source_type == "video":
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(source_data.read())
        cap = cv2.VideoCapture(tfile_in.name)
        ret, frame = cap.read()
        cap.release()
    elif source_type == "gif":
        gif_img = Image.open(source_data)
        gif_img.seek(0)
        frame_rgba = gif_img.convert("RGBA")
        bg = Image.new("RGBA", frame_rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(frame_rgba)
        frame = cv2.cvtColor(np.array(bg), cv2.COLOR_RGBA2BGR)
    elif source_type == "sequence":
        source_data = sorted(source_data, key=lambda x: x.name)
        file_bytes = np.asarray(bytearray(source_data[0].read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    return frame

def process_single_image(image_file, fps, duration, intensity, scale, export_fmt):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    fg_cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    h, w = fg_cv_image.shape[:2]
    total_frames = int(fps * duration)
    
    updates_per_sec = 12
    frames_per_update = max(1, int(fps / updates_per_sec))
    
    map_x_base, map_y_base = np.meshgrid(np.arange(w), np.arange(h))
    map_x_base = map_x_base.astype(np.float32)
    map_y_base = map_y_base.astype(np.float32)

    progress_bar = st.progress(0)
    current_map_x, current_map_y = None, None
    processed_frames = []

    for i in range(total_frames):
        if i % frames_per_update == 0 or current_map_x is None:
            current_map_x, current_map_y = generate_noise_map(w, h, scale, intensity, map_x_base, map_y_base)
            
        frame = cv2.remap(fg_cv_image, current_map_x, current_map_y, 
                          interpolation=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REPLICATE)
        
        frame = composite_on_white(frame)
        processed_frames.append(frame)
        progress_bar.progress((i + 1) / total_frames)
        
    return export_frames(processed_frames, fps, export_fmt, w, h)

def process_animation(source_type, source_data, out_fps, intensity, scale, export_fmt):
    processed_frames = []
    cap = None
    gif_img = None
    
    if source_type == "video":
        tfile_in = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile_in.write(source_data.read())
        cap = cv2.VideoCapture(tfile_in.name)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    elif source_type == "gif":
        gif_img = Image.open(source_data)
        total_frames = gif_img.n_frames
        w, h = gif_img.size
    elif source_type == "sequence":
        source_data = sorted(source_data, key=lambda x: x.name)
        total_frames = len(source_data)
        first_frame = cv2.imdecode(np.asarray(bytearray(source_data[0].read()), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        h, w = first_frame.shape[:2]
        source_data[0].seek(0)

    updates_per_sec = 12
    frames_per_update = max(1, int(out_fps / updates_per_sec))
    
    map_x_base, map_y_base = np.meshgrid(np.arange(w), np.arange(h))
    map_x_base, map_y_base = map_x_base.astype(np.float32), map_y_base.astype(np.float32)
    
    current_map_x, current_map_y = None, None
    progress_bar = st.progress(0)
    
    for frame_count in range(total_frames):
        frame = None
        if source_type == "video":
            ret, frame = cap.read()
            if not ret: break
        elif source_type == "gif":
            gif_img.seek(frame_count)
            frame_rgba = gif_img.convert("RGBA")
            bg = Image.new("RGBA", frame_rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(frame_rgba)
            frame = cv2.cvtColor(np.array(bg), cv2.COLOR_RGBA2BGR)
        elif source_type == "sequence":
            file_bytes = np.asarray(bytearray(source_data[frame_count].read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if frame is None: break
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            frame = composite_on_white(frame)

        if frame_count % frames_per_update == 0 or current_map_x is None:
             current_map_x, current_map_y = generate_noise_map(w, h, scale, intensity, map_x_base, map_y_base)
             
        distorted_frame = cv2.remap(frame, current_map_x, current_map_y, 
                                  interpolation=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_REPLICATE)
        processed_frames.append(distorted_frame)
        
        if total_frames > 0:
            progress_bar.progress(min((frame_count + 1) / total_frames, 1.0))
            
    if cap: cap.release()
    return export_frames(processed_frames, out_fps, export_fmt, w, h)

# --- UI Layout ---
tab1, tab2 = st.tabs(["Single Image", "Video / Animation"])

with tab1:
    st.header("Single Image Animation")
    uploaded_img = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    preview_container_single = st.empty()
    
    col1, col2 = st.columns(2)
    with col1:
        s_intensity = st.slider("Wobble Intensity", 1, 20, 1, key="s_int")
        s_fps = st.number_input("FPS", value=24, key="s_fps")
        s_export = st.selectbox("Export Format", ["MP4 Video", "GIF Animation", "PNG Sequence (ZIP)"], key="s_exp")
    with col2:
        s_scale = st.slider("Wave Scale", 5, 100, 20, key="s_sc")
        s_dur = st.number_input("Duration (sec)", value=3, key="s_dur")
        
    btn_col1, btn_col2 = st.columns(2)
    
    if btn_col1.button("Preview Effect", use_container_width=True, disabled=not uploaded_img):
        with st.spinner("Generating 2s test loop..."):
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            fg_cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            uploaded_img.seek(0) # Reset file pointer for the generator
            
            preview_gif = generate_quick_preview(fg_cv_image, s_intensity, s_scale)
            preview_container_single.image(preview_gif, caption="Live Preview (12 FPS)")
            
    if btn_col2.button("Generate Final Output", type="primary", use_container_width=True, disabled=not uploaded_img):
        with st.spinner("Processing full sequence..."):
            out_path, mime_type, file_name = process_single_image(uploaded_img, s_fps, s_dur, s_intensity, s_scale, s_export)
            
            if s_export == "MP4 Video":
                st.video(out_path)
            elif s_export == "GIF Animation":
                st.image(out_path)
            else:
                st.success("Sequence successfully zipped!")
                
            with open(out_path, 'rb') as f:
                st.download_button("Download Output", f, file_name=file_name, mime=mime_type, use_container_width=True)


with tab2:
    st.header("Animation Source")
    
    source_opt = st.radio("Source Type", ["Video / GIF File", "Image Sequence (Multiple Files)"])
    
    if source_opt == "Video / GIF File":
        uploaded_vid = st.file_uploader("Upload Video/GIF", type=['mp4', 'mov', 'avi', 'gif'])
        source_type = "gif" if uploaded_vid and uploaded_vid.name.lower().endswith(".gif") else "video"
        vid_data = uploaded_vid
    else:
        uploaded_vid = st.file_uploader("Upload Image Sequence", type=['png', 'jpg', 'jpeg', 'bmp'], accept_multiple_files=True)
        source_type = "sequence"
        vid_data = uploaded_vid if len(uploaded_vid) > 0 else None
        if vid_data: st.info(f"{len(vid_data)} images loaded.")

    preview_container_vid = st.empty()

    st.subheader("Wobble Physics")
    col3, col4 = st.columns(2)
    with col3:
        v_intensity = st.slider("Wobble Intensity", 1, 20, 1, key="v_int")
        v_fps = st.number_input("Output FPS", value=24, key="v_fps")
    with col4:
        v_scale = st.slider("Wave Scale", 5, 100, 20, key="v_sc")
        v_export = st.selectbox("Export Format", ["MP4 Video", "GIF Animation", "PNG Sequence (ZIP)"], key="v_exp")
        
    btn_col3, btn_col4 = st.columns(2)
    
    if btn_col3.button("Preview Effect", use_container_width=True, disabled=not vid_data):
        with st.spinner("Grabbing first frame and running test loop..."):
            first_frame = get_first_frame(source_type, vid_data)
            
            # Reset file pointers so the final generator doesn't miss the first frame
            if source_type == "sequence":
                vid_data[0].seek(0)
            else:
                vid_data.seek(0)
                
            if first_frame is not None:
                preview_gif = generate_quick_preview(first_frame, v_intensity, v_scale)
                preview_container_vid.image(preview_gif, caption="Preview Effect on First Frame (12 FPS)")
                
    if btn_col4.button("Process Full Animation", type="primary", use_container_width=True, disabled=not vid_data):
        with st.spinner("Processing entire video..."):
            out_path, mime_type, file_name = process_animation(source_type, vid_data, v_fps, v_intensity, v_scale, v_export)
            
            if v_export == "MP4 Video":
                st.video(out_path)
            elif v_export == "GIF Animation":
                st.image(out_path)
            else:
                st.success("Sequence successfully zipped!")
                
            with open(out_path, 'rb') as f:
                st.download_button("Download Output", f, file_name=file_name, mime=mime_type, use_container_width=True)
