import streamlit as st
import base64
import os
import cv2
import numpy as np
import tempfile
import uuid
import zipfile
from PIL import Image


class Equirectangular:
    def __init__(self, img_name):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape

    def GetPerspective(self, FOV, THETA, PHI, height, width, RADIUS=128):
        equ_h = self._height
        equ_w = self._width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = FOV
        hFOV = float(height) / width * wFOV

        c_x = (width - 1) / 2.0
        c_y = (height - 1) / 2.0

        wangle = (180 - wFOV) / 2.0
        w_len = 2 * RADIUS * np.sin(np.radians(wFOV / 2.0)) / np.sin(np.radians(wangle))
        w_interval = w_len / (width - 1)

        hangle = (180 - hFOV) / 2.0
        h_len = 2 * RADIUS * np.sin(np.radians(hFOV / 2.0)) / np.sin(np.radians(hangle))
        h_interval = h_len / (height - 1)
        x_map = np.zeros([height, width], np.float32) + RADIUS
        y_map = np.tile((np.arange(0, width) - c_x) * w_interval, [height, 1])
        z_map = -np.tile((np.arange(0, height) - c_y) * h_interval, [width, 1]).T
        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.zeros([height, width, 3], float)
        xyz[:, :, 0] = (RADIUS / D * x_map)[:, :]
        xyz[:, :, 1] = (RADIUS / D * y_map)[:, :]
        xyz[:, :, 2] = (RADIUS / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / RADIUS)
        lon = np.zeros([height * width], float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([height, width]) / np.pi * 180
        lat = -lat.reshape([height, width]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        persp = cv2.remap(self._img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return persp

def slice_panorama(image_path, fov, theta_step, phi, height, width, output_dir):
    equ = Equirectangular(image_path)
    sliced_images = []

    for theta in range(0, 360, theta_step):
        img = equ.GetPerspective(fov, theta, phi, height, width)
        img_path = os.path.join(output_dir, f"{theta:03d}.jpg")
        cv2.imwrite(img_path, img)
        sliced_images.append(img_path)

    return sliced_images

def create_video(sliced_images, fps, output_dir):
    video_path = os.path.join(output_dir, "fullvideo.mp4")

    frame_width = cv2.imread(sliced_images[0]).shape[1]
    frame_height = cv2.imread(sliced_images[0]).shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    for image_path in sliced_images:
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()

    return video_path

def zip_folder(folder_path):
    zip_path = os.path.join(tempfile.gettempdir(), "panorama_processing.zip")
    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zip_file.write(file_path, arcname)
    return zip_path

def process_single_image(image_path, settings_list):
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_base_dir = os.path.join(tempfile.gettempdir(), base_name)
    os.makedirs(output_base_dir, exist_ok=True)
    
    all_sliced_images = []

    for settings in settings_list:
        fov, theta_step, phi, height, width, fps = settings
        folder_name = f"{base_name}_{fov}_{theta_step}_{phi}_{fps}"
        output_dir = os.path.join(output_base_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        sliced_images = slice_panorama(image_path, fov, theta_step, phi, height, width, output_dir)
        create_video(sliced_images, fps, output_dir)
        all_sliced_images.extend(sliced_images)
    
    zip_path = zip_folder(output_base_dir)
    return all_sliced_images, zip_path, output_base_dir

def process_folder(folder_path, settings_list):
    output_base_dir = os.path.join(tempfile.gettempdir(), os.path.basename(folder_path) + str(uuid.uuid4()))
    os.makedirs(output_base_dir, exist_ok=True)
    all_sliced_images = []

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(folder_path, filename)
            base_name = os.path.splitext(filename)[0]
            for settings in settings_list:
                fov, theta_step, phi, height, width, fps = settings
                folder_name = f"{base_name}_{fov}_{theta_step}_{phi}_{fps}"
                image_output_dir = os.path.join(output_base_dir, folder_name)
                os.makedirs(image_output_dir, exist_ok=True)
                
                sliced_images = slice_panorama(image_path, fov, theta_step, phi, height, width, image_output_dir)
                create_video(sliced_images, fps, image_output_dir)
                all_sliced_images.extend(sliced_images)
    
    zip_path = zip_folder(output_base_dir)
    return all_sliced_images, zip_path, output_base_dir

st.set_page_config(page_title="360° Image Processing", layout="wide")
st.title("360° Image Processing Tool")

tab1, tab2 = st.tabs(["Single Image Slicer", "Folder Image Slicer"])

with tab1:
    st.header("Single Image Panorama Slicer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Panorama Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            # 업로드된 이미지 표시
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # OpenCV로 이미지를 읽어서 확인
            test_img = cv2.imread(temp_path)
            if test_img is None:
                st.error("이미지를 읽을 수 없습니다. 다른 이미지를 시도해주세요.")
                st.stop()
            
            fov = st.slider("FOV", 30, 150, 90, 5)
            theta_step = st.slider("Theta Step", 1, 45, 5, 1)
            st.markdown("Theta Step: The step size for the left/right angle (in degrees)")
            phi = st.slider("Phi", -90, 90, 0, 5)
            st.markdown("Phi: The up/down angle (in degrees)")
            height = st.slider("Slice Image Height", 240, 1080, 720, 1)
            width = st.slider("Slice Image Width", 320, 1920, 1280, 1)
            fps = st.slider("Video FPS", 1, 60, 3, 1)
            
            if 'settings_list' not in st.session_state:
                st.session_state.settings_list = []
            
            if st.button("Add Settings"):
                st.session_state.settings_list.append((fov, theta_step, phi, height, width, fps))
                settings_text = "\n".join([
                    f"FOV: {s[0]}, Theta Step: {s[1]}, Phi: {s[2]}, Height: {s[3]}, Width: {s[4]}, FPS: {s[5]}" 
                    for s in st.session_state.settings_list
                ])
                st.text_area("Current Settings", value=settings_text, height=150)
            
            if st.button("Process Single Image"):
                with st.spinner("Processing image..."):
                    sliced_images, zip_path, output_dir = process_single_image(
                        temp_path, 
                        st.session_state.settings_list
                    )
                    
                    st.success("Processing complete!")
                    
                    with open(zip_path, "rb") as f:
                        st.download_button(
                            "Download Processed Images and Videos",
                            f,
                            file_name="panorama_processing.zip",
                            mime="application/zip"
                        )
    
    with col2:
        if 'sliced_images' in locals():
            st.subheader("Processed Images")
            for img_path in sliced_images[:10]:  # 처음 10개 이미지만 표시
                img = Image.open(img_path)
                st.image(img, use_column_width=True)

with tab2:
    st.header("Folder Image Panorama Slicer")
    
    folder_path = st.text_input("Enter Folder Path", "")
    
    if folder_path and os.path.exists(folder_path):
        fov = st.slider("FOV (Folder)", 30, 150, 90, 5)
        theta_step = st.slider("Theta Step (Folder)", 1, 45, 5, 1)
        phi = st.slider("Phi (Folder)", -90, 90, 0, 5)
        height = st.slider("Slice Image Height (Folder)", 240, 1080, 720, 1)
        width = st.slider("Slice Image Width (Folder)", 320, 1920, 1280, 1)
        fps = st.slider("Video FPS (Folder)", 1, 60, 3, 1)
        
        if 'folder_settings_list' not in st.session_state:
            st.session_state.folder_settings_list = []
        
        if st.button("Add Settings (Folder)"):
            st.session_state.folder_settings_list.append((fov, theta_step, phi, height, width, fps))
            settings_text = "\n".join([
                f"FOV: {s[0]}, Theta Step: {s[1]}, Phi: {s[2]}, Height: {s[3]}, Width: {s[4]}, FPS: {s[5]}" 
                for s in st.session_state.folder_settings_list
            ])
            st.text_area("Current Settings (Folder)", value=settings_text, height=150)
        
        if st.button("Process Folder"):
            with st.spinner("Processing folder..."):
                sliced_images, zip_path, output_dir = process_folder(
                    folder_path, 
                    st.session_state.folder_settings_list
                )
                
                st.success("Folder processing complete!")
                
                with open(zip_path, "rb") as f:
                    st.download_button(
                        "Download Processed Images and Videos (Folder)",
                        f,
                        file_name="panorama_processing_folder.zip",
                        mime="application/zip"
                    )