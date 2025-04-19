import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.utils.court_mapper import CourtMapper
import tempfile
import os
from collections import deque
import time


class CourtVideoProcessor:
    def __init__(self, video_path, court_part="left", buffer_size=5, batch_size = 8, process_nth_frame=2, progress_callback=None, locally=False):
        self.video_path = video_path
        self.court_part = court_part
        self.mapper = CourtMapper(locally=locally)
        self.court_image = cv2.imread("assets/court_diagram.jpeg")
        self.court_image = cv2.cvtColor(self.court_image, cv2.COLOR_BGR2RGB)
        self.prev_player_positions = None
        self.position_buffer = deque(maxlen=buffer_size)  # Store previous positions for smoothing
        self.process_nth_frame = process_nth_frame  # Process every nth frame
        self.batch_size = batch_size
        self.progress_callback = progress_callback
        
    def smooth_positions(self, positions):
        """Apply smoothing to player positions using moving average"""
        if not positions:
            return positions
        
        # Add current positions to buffer
        self.position_buffer.append(positions)
        
        if len(self.position_buffer) < 2:
            return positions
            
        # Create smoothed positions
        smoothed = []
        # Get all positions from buffer
        all_positions = list(self.position_buffer)
        
        # Track players across frames and smooth their positions
        for current_pos in positions:
            positions_history = []
            curr_x, curr_y, label, team = current_pos
            
            # Find position history for this player
            for past_positions in all_positions[:-1]:
                closest_pos = min(past_positions, 
                                key=lambda p: np.sqrt((p[0]-curr_x)**2 + (p[1]-curr_y)**2)
                                if p[2] == label else float('inf'))
                if closest_pos[2] == label:
                    positions_history.append((closest_pos[0], closest_pos[1]))
            
            # Calculate smoothed position
            if positions_history:
                positions_history.append((curr_x, curr_y))
                smooth_x = sum(p[0] for p in positions_history) / len(positions_history)
                smooth_y = sum(p[1] for p in positions_history) / len(positions_history)
                smoothed.append((int(smooth_x), int(smooth_y), label, team))
            else:
                smoothed.append(current_pos)
                
        return smoothed


    def process_batch(self, frames):
        """Process a batch of frames and return player positions"""
        # Save each frame to a temporary file
        temp_paths = []
        try:
            for frame in frames:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                    temp_path = temp_file.name
                    cv2.imwrite(temp_path, frame)
                    temp_paths.append(temp_path)
            
            # Process frames in a batch
            all_player_positions = self.mapper.detect_and_map_players_batch(temp_paths, self.court_part)
            
            # Apply smoothing to each frame's positions
            for i, player_positions in enumerate(all_player_positions):
                if player_positions:
                    # Apply smoothing to positions
                    all_player_positions[i] = self.smooth_positions(player_positions)
                    self.prev_player_positions = all_player_positions[i]
                else:
                    all_player_positions[i] = self.prev_player_positions
            
            return all_player_positions
            
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            return [self.prev_player_positions] * len(frames)
        finally:
            # Clean up temporary files
            for temp_path in temp_paths:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def process_video(self, output_path=None, display=True):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open input video")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(self.court_image.shape[1])
        frame_height = int(self.court_image.shape[0])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        # Adjust display refresh rate
        display_interval = 1/30  # 30 FPS display

        
        if display:
            plt.ion()
            fig, ax = plt.subplots(figsize=(10, 8))

        last_display_time = 0
        batch_frames = []
        batch_indices = []
        final_frames = []
        
        if output_path:
            # Use more compatible codec and ensure frame size is valid
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps//self.process_nth_frame,
                (frame_width, frame_height),
                isColor=True
            )
            
            # Verify video writer is initialized
            if not out.isOpened():
                cap.release()
                raise RuntimeError("Failed to initialize video writer")
        
        try:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if self.progress_callback:
                    self.progress_callback(frame_count)
                
                # Process every nth frame
                if frame_count % self.process_nth_frame == 0:

                    batch_frames.append(frame)
                    batch_indices.append(frame_count)

                if len(batch_frames) >= self.batch_size or (frame_count + 1 >= total_frames):

                    start_time = time.time()
                    player_positions_batch = self.process_batch(batch_frames)
                    print(f"Processed batch of {len(batch_frames)} frames in {time.time() - start_time:.2f} seconds")

                    for i, (positions, idx) in enumerate(zip(player_positions_batch, batch_indices)):    
                        court_frame = self.court_image.copy()
                    
                        if positions:
                            self.mapper.update_metrics(positions, idx)
                            for position in positions:
                                x, y, det_label, team = position

                                if det_label == 0:
                                    if team == 0:
                                        color = (255, 0, 0)
                                        size = 8
                                    else:
                                        color = (0, 0, 255)
                                        size = 8
                                else:
                                    color = (255, 255, 255)
                                    size = 4
                                # Draw player marker
                                cv2.circle(court_frame, (int(x), int(y)), size, color, -1)
                                # Add outline for better visibility
                                cv2.circle(court_frame, (int(x), int(y)), size+2, (255, 255, 255), 2)
                        
                        current_time = time.time()
                        if display and (current_time - last_display_time) >= display_interval:
                            ax.clear()
                            ax.imshow(court_frame)
                            # Add legend
                            legend_elements = [
                                plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=(1, 0, 0), markersize=10, label='Team 1'),
                                plt.Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=(0, 0, 1), markersize=10, label='Team 2')
                            ]
                            ax.legend(handles=legend_elements, loc='upper right')
                            ax.axis('off')
                            plt.pause(0.001)
                            last_display_time = current_time
                        
                        # Convert to BGR for video writing
                        out_frame = cv2.cvtColor(court_frame, cv2.COLOR_RGB2BGR)
                        final_frames.append(out_frame)
                    batch_frames = []
                    batch_indices = []
                frame_count += 1
        
                        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
        finally:
            cap.release()
            if output_path and 'out' in locals():
                for frame in final_frames:
                    out.write(frame)
                out.release()
                return self.mapper.get_metrics_summary()
            if display:
                plt.ioff()
                plt.close()

    def process_frame(self, frame):
        """Process a single frame and return player positions (for backward compatibility)"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, frame)
        
        try:
            player_positions = self.mapper.detect_and_map_players(temp_path, self.court_part)
            if player_positions:
                # Apply smoothing to positions
                smoothed_positions = self.smooth_positions(player_positions)
                self.prev_player_positions = smoothed_positions
                return smoothed_positions
            return self.prev_player_positions
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return self.prev_player_positions
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

