from time import sleep
from asset_handler import ThugAsset, CigaAsset
import cv2
import face_alignment
import torch
import numpy as np
import pyvirtualcam
from threading import Event, Thread
from config import face_alignement_conf, vcamera_conf

# Read images to be placed on the output stram
thug = ThugAsset("./assets/thug.png")
ciga = CigaAsset("./assets/ciga.png")

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=face_alignement_conf['device'], flip_input=True,
                                  face_detector=face_alignement_conf['blazeface']['face_detector'],
                                  face_detector_kwargs=face_alignement_conf['blazeface']['face_detector_kwargs'])
cam = cv2.VideoCapture(0)

initial_log_flag = True

stopper_event = Event()


def stopper_fn():
    stopper_event.set()
    # dirty crutch to make console output consistent
    sleep(10)
    while True:
        q = input(">>> input `q` to quit: ")
        print(q)
        if q == 'q':
            print(">>> exiting stopper...")
            stopper_event.clear()
            return


def main_loop_fn():
    with pyvirtualcam.Camera(**vcamera_conf) as vcam:
        print(f'>>> Using virtual camera: {vcam.device}')
        while stopper_event.is_set():
            ret, frame = cam.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(float) / 255
            faces = None
            try:
                faces = fa.get_landmarks_from_image(torch.tensor(frame_rgb))
            except Exception as e:
                print(f">>> Failed to retrieve predictions: {e}")

            if faces:
                # draw assets
                for face in faces:
                    frame = thug.apply_asset(frame, face)
                    frame = ciga.apply_asset(frame, face)

            global initial_log_flag
            if initial_log_flag:
                print(">>> Stream started...")
                initial_log_flag = False

            vcam.send(cv2.cvtColor(
                (frame*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            vcam.sleep_until_next_frame()

            # cv2.imshow("frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        print(">>> exiting main loop...")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    stopper_thread = Thread(target=stopper_fn)
    main_loop_thread = Thread(target=main_loop_fn)
    stopper_thread.start()
    main_loop_thread.start()
