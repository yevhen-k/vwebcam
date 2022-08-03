vcamera_conf = {
    'width': 640,
    'height': 480,
    'fps': 20,
    'backend': 'v4l2loopback',
    'device': '/dev/video2',
}

face_alignement_conf = {
    'blazeface': {
        'face_detector': 'blazeface',
        'face_detector_kwargs': {},
    },
    'sfd': {
        'face_detector': 'sfd',
        'face_detector_kwargs': {
            'filter_threshold': 0.8,
        },
    },
    'device': 'cuda',
}
