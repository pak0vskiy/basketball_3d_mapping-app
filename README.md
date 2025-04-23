# Basketball Court 3D Mapping Application

A real-time basketball analytics tool that maps player positions from video/image input onto a 2D court diagram, tracks team movements, and generates advanced game metrics using computer vision.

## Features

- **Multi-Format Input Support**
  - Single image processing
  - Video processing with real-time mapping
  - Supports common formats (jpg, jpeg, png, mp4, avi, mov)

- **Player Detection & Tracking**
  - Automatic player detection using YOLO
  - Team classification based on jersey colors
  - Real-time position mapping to 2D court diagram

- **Advanced Analytics**
  - Team-based heatmaps
  - Distance covered tracking
  - Court coverage analysis
  - Both half-court sides support (left/right)

- **Performance Optimization**
  - GPU acceleration support
  - Batch processing for videos
  - Configurable processing parameters
  - Frame processing optimization

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- FFmpeg

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/basketball_3dMapping_app.git
cd basketball_3dMapping_app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```env
PLAYER_API_URL=your_player_detection_api_url
COURTSEG_API_URL=your_court_segmentation_api_url
HUGGING_FACE_KEY=your_huggingface_token
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. In the web interface:
   - Choose between image or video input
   - Select court side (left/right)
   - Upload your basketball game footage
   - Adjust processing parameters if needed
   - View results in real-time

### Video Processing Parameters

- **Smoothing Buffer Size**: Adjust tracking smoothness (5-30)
- **Process Every Nth Frame**: Balance between speed and accuracy (1-5)
- **Batch Size**: Optimize GPU utilization (1-64)

## Project Structure

```
basketball_3dMapping_app/
├── app.py                 # Main Streamlit application
├── src/
│   └── utils/
│       ├── court_mapper.py       # Court mapping logic
│       ├── video_processor.py    # Video processing
│       └── find_reference_points.py  # Court point detection
├── assets/
│   └── court_diagram.jpeg    # Court template
├── model/                    # Model weights
└── ffmpeg/                   # FFmpeg binaries
```

## Technical Details

- **Models**: YOLO-based architecture for player detection and court segmentation
- **Computer Vision**: OpenCV for image processing and homography
- **UI**: Streamlit for interactive web interface
- **Video Processing**: FFmpeg for video handling
- **Team Classification**: K-means clustering for jersey color analysis

## License

[MIT LICENCE]

## Acknowledgments

- YOLOv11 for player detection
- YOLOv8 for court segmentation
- Streamlit for the web interface
- OpenCV for computer vision operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.