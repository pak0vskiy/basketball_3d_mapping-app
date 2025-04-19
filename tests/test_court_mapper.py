basketball_court_keypoints_detection/
│
├── datasets/
│   ├── court_keyPoints/
│   │   ├── images/
│   │   │   ├── test/
│   │   │   │   ├── 1.jpg
│   │   │   │   ├── 2.jpg
│   │   │   │   └── 3.jpg
│   │   │   └── train/
│   │   │       ├── 1.jpg
│   │   │       ├── 2.jpg
│   │   │       └── 3.jpg
│   │   └── masks/
│   │       ├── test/
│   │       └── train/
│   └── videos/
│       ├── test_video/
│       │   └── 0.0-17.0.mp4
│       └── train_video/
│           └── 0.0-17.0.mp4
│
├── models/
│   ├── player_detection_model.pt
│   └── court_segmentation.pt
│
├── utils/
│   ├── __init__.py
│   ├── court_mapper.py
│   ├── find_reference_points.py
│   └── video_processor.py
│
├── notebooks/
│   ├── homography_transformation.ipynb
│   └── data_analysis.ipynb
│
├── requirements.txt
├── README.md
└── main.py