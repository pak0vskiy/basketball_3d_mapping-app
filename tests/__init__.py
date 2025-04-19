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
│   │   │       ├── train1.jpg
│   │   │       └── train2.jpg
│   │   └── masks/
│   │       ├── mask1.png
│   │       └── mask2.png
│   └── test_video/
│       └── 0.0-17.0.mp4
│
├── model/
│   ├── court_segmentation.pt
│   └── player_detection_model.pt
│
├── utils/
│   ├── __init__.py
│   ├── court_mapper.py
│   ├── find_reference_points.py
│   └── video_processor.py
│
├── notebooks/
│   ├── homography_transformation.ipynb
│   └── data_visualization.ipynb
│
├── requirements.txt
├── README.md
└── main.py