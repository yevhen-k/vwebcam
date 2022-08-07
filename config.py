vcamera_conf = {
    'width': 640,
    'height': 480,
    'fps': 20,
    'backend': 'v4l2loopback',
    'device': '/dev/video2',
}

face_mesh_conf = {
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}