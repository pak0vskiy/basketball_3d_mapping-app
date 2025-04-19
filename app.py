import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.utils.court_mapper import CourtMapper
from src.utils.video_processor import CourtVideoProcessor
import subprocess
import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt

def check_gpu():
    if torch.cuda.is_available():
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
        gpu_memory = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        st.success(f"✅ Using GPU - {gpu_info}, {gpu_memory}")
        return True
    else:
        st.warning("❌ No GPU detected - using CPU for inference")
        return False

def display_metrics(metrics, court_side="left"):
    st.subheader("Game Metrics")
    
    # Stats columns
    col1, col2 = st.columns(2)
    with col1:
        st.write("Team 1 (Red)")
        st.write(f"Distance covered: {metrics[0]['total_distance_km']} km")
        st.write(f"Ball possession: {metrics[0]['possession_percentage']}%")
        st.write(f"Court coverage: {metrics[0]['court_coverage_percentage']}%")
        
        
    with col2:
        st.write("Team 2 (Blue)")  
        st.write(f"Distance covered: {metrics[1]['total_distance_km']} km")
        st.write(f"Ball possession: {metrics[1]['possession_percentage']}%")
        st.write(f"Court coverage: {metrics[1]['court_coverage_percentage']}%")
        
    if 'heatmap' in metrics[0] and 'heatmap' in metrics[1]:
        st.subheader("Team Movement Heatmaps")
        
        # Load court image with correct dimensions
        court_image = cv2.imread("assets/court_diagram.jpeg")
        court_image = cv2.cvtColor(court_image, cv2.COLOR_BGR2RGB)
        
        # Calculate the half-court boundaries
        court_width = 640  # Full court width
        court_height = 372  # Full court height
        mid_x = court_width // 2
        
        # Set display boundaries based on court side
        if court_side == "left":
            court_image = court_image[:, :mid_x]
            court_bounds = [0, mid_x, court_height, 0]  # Flip Y-axis for matplotlib
        else:
            court_image = court_image[:, mid_x:]
            court_bounds = [mid_x, court_width, court_height, 0]  # Flip Y-axis for matplotlib
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot cropped court image
        ax.imshow(court_image, extent=court_bounds)
        
        # Use original heatmaps without resizing
        heatmap1 = metrics[0]['heatmap']
        heatmap2 = metrics[1]['heatmap']
        
        # Crop heatmaps based on selected court side
        if court_side == "left":
            heatmap1 = heatmap1[:, :mid_x]
            heatmap2 = heatmap2[:, :mid_x]
        else:
            heatmap1 = heatmap1[:, mid_x:]
            heatmap2 = heatmap2[:, mid_x:]
        
        # Apply Gaussian blur to smooth the heatmaps
        heatmap1 = cv2.GaussianBlur(heatmap1, (15, 15), 0)
        heatmap2 = cv2.GaussianBlur(heatmap2, (15, 15), 0)
        
        # Normalize heatmaps
        max_value1 = np.max(heatmap1)
        max_value2 = np.max(heatmap2)
        
        if max_value1 > 0:
            heatmap1 = np.clip(heatmap1 / max_value1 * 5, 0, 1)
        if max_value2 > 0:
            heatmap2 = np.clip(heatmap2 / max_value2 * 5, 0, 1)
        
        # Overlay team heatmaps with transparency
        orange_heatmap = ax.imshow(heatmap1,
                                 cmap='Reds',
                                 alpha=0.6,
                                 extent=court_bounds,
                                 origin='upper')  # Match image orientation
        
        blue_heatmap = ax.imshow(heatmap2,
                                cmap='Blues',
                                alpha=0.6,
                                extent=court_bounds,
                                origin='upper')  # Match image orientation
        
        # Add colorbars
        orange_colorbar = plt.colorbar(orange_heatmap, ax=ax, label='Team 1 Presence', 
                                     orientation='horizontal', pad=0.01)
        blue_colorbar = plt.colorbar(blue_heatmap, ax=ax, label='Team 2 Presence',
                                   orientation='horizontal', pad=0.05)
        
        # Remove axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title
        ax.set_title(f"Team Movement Heatmaps ({court_side.capitalize()} Court)")
        
        # Display the plot
        st.pyplot(fig)

def main():
    st.title("Basketball Court Mapping")
    st.sidebar.title("Settings")

    locally = st.selectbox("Run locally?", [True, False])

    if locally:
        mapper = CourtMapper(locally=True)
        using_gpu = check_gpu()
    else:
        mapper = CourtMapper(locally=False)
    

    
    # Add file uploader that accepts both images and videos
    upload_type = st.sidebar.selectbox("Select Input Type", ["Image", "Video"])
    
    if upload_type == "Image":
        uploaded_file = st.file_uploader("Upload a basketball court image", type=['jpg', 'jpeg', 'png'])
        court_side = st.sidebar.selectbox("Select Court Side", ["left", "right"])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_file.read())
            
            # Display original image
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(uploaded_file)
            
            # Process image and display results
            with col2:
                st.subheader("Mapped Court")
                # Create court diagram
                court_image = cv2.imread("assets/court_diagram.jpeg")
                court_image = cv2.cvtColor(court_image, cv2.COLOR_BGR2RGB)
                
                # Detect and map players
                player_positions = mapper.detect_and_map_players("temp_image.jpg", court_side)
                
                # Draw players on court
                for point in player_positions:
                    x, y, det_label, team = point
                    if det_label == 0:  # Players
                        if team == 0:
                            color = (255, 165, 0)  # Orange for team 1
                            size = 8
                        else:
                            color = (0, 0, 255)    # Blue for team 2
                            size = 8
                    else:  # Ball
                        color = (255, 255, 255)  # White for ball
                        size = 4
                    
                    # Draw marker with white outline
                    cv2.circle(court_image, (int(x), int(y)), size+2, (255, 255, 255), 2)
                    cv2.circle(court_image, (int(x), int(y)), size, color, -1)
                
                st.image(court_image)
    

    else:  # Video processing
        uploaded_file = st.file_uploader("Upload a basketball court video", type=['mp4', 'avi', 'mov'])
        court_side = st.sidebar.selectbox("Select Court Side", ["left", "right"])
        buffer_size = st.sidebar.slider("Smoothing Buffer Size", 5, 30, 15)
        process_nth_frame = st.sidebar.slider("Process Every Nth Frame", 1, 5, 2)
        input_batch_size = st.sidebar.slider("Batch size", 1, 64, 16)
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.read())
            
            temp_video = cv2.VideoCapture("temp_video.mp4")
            total_frames = int(temp_video.get(cv2.CAP_PROP_FRAME_COUNT))
            temp_video.release()
            
            if st.button("Process Video"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    with st.spinner("Processing video... This may take a while."):
                        def progress_callback(frame_num):
                            progress = int((frame_num / total_frames) * 100)
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {frame_num} of {total_frames} ({progress}%)")

                        # Initialize and run video processor
                        processor = CourtVideoProcessor(
                            "temp_video.mp4",
                            court_part=court_side,
                            buffer_size=buffer_size,
                            process_nth_frame=process_nth_frame,
                            batch_size=input_batch_size,
                            progress_callback=progress_callback,
                            locally=locally
                        )
                        
                        # Process video
                        metrics = processor.process_video(output_path="mapped_output.mp4", display=False)
                        # Ensure the video writer is fully closed
                        cv2.destroyAllWindows()
                        
                        # Convert video format
                        ffmpeg_path = os.path.join("ffmpeg", "bin", "ffmpeg.exe")
                        subprocess.run([
                            ffmpeg_path, "-y",
                            "-i", "mapped_output.mp4",
                            "-c:v", "libx264",
                            "-preset", "medium",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            "output_final.mp4"
                        ], check=True, capture_output=True)
                        
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Show success message
                        st.success("Video processing complete!")
                        
                        # Display videos
                        st.subheader("Original Video")
                        st.video("temp_video.mp4")
                        st.subheader("Mapped Video")
                        st.video("output_final.mp4")
                        display_metrics(metrics, court_side)
                        
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
            



if __name__ == "__main__":
    main()