import os
from time import sleep
import json
from typing import Dict
from asset_handler import ThugAsset, CigaAsset
import cv2
import face_alignment
import torch
import numpy as np
import pyvirtualcam
from threading import Event, Lock, Thread
from config import face_alignement_conf, vcamera_conf
from utils import StateChecker

# Read images to be placed on the output stram
thug = ThugAsset("./assets/thug.png")
ciga = CigaAsset("./assets/ciga.png")


def stopper_fn(stopper_event: Event) -> None:
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


def state_checker_fn(state_checker: StateChecker, stopper_event: Event) -> None:
    state_checker.disable_content_access()
    while stopper_event.is_set():
        sleep(state_checker.sleep_duration_sec)
        if state_checker.is_state_changed():
            state_checker.enable_content_access()


def main_loop_fn(camera: cv2.VideoCapture, model: face_alignment.FaceAlignment,
                 stopper_event: Event,
                 state_checker: StateChecker) -> None:
    initial_log_flag = True
    enabler = state_checker.content
    with pyvirtualcam.Camera(**vcamera_conf) as vcam:
        print(f'>>> Using virtual camera: {vcam.device}')
        while stopper_event.is_set():

            # get hot reload state
            if state_checker.is_content_available():
                enabler = state_checker.get_actual_content()
                state_checker.disable_content_access()
                print("\nconfig changed\n>>>")

            success, frame = camera.read()

            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(float) / 255
            faces = None
            if enabler["enable_dnn"]:
                try:
                    faces = fa.get_landmarks_from_image(
                        torch.tensor(frame_rgb))
                except Exception as e:
                    print(f">>> Failed to retrieve predictions: {e}")

            if faces:
                # draw assets
                for face in faces:
                    if enabler["enable_thug"]:
                        frame = thug.apply_asset(frame, face)
                    if enabler["enable_ciga"]:
                        frame = ciga.apply_asset(frame, face)

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

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=face_alignement_conf['device'], flip_input=True,
                                      face_detector=face_alignement_conf['blazeface']['face_detector'],
                                      face_detector_kwargs=face_alignement_conf['blazeface']['face_detector_kwargs'])
    cam = cv2.VideoCapture(0)

    stopper_event = Event()
    stopper_event.set()

    # Prepare dirty hot-reload config
    enabler_path = "./enabler.json"
    state_checker_event = Event()
    state_checker_event_lock = Lock()
    state_checker = StateChecker(
        enabler_path, state_checker_event, state_checker_event_lock, sleep_duration_sec=1)

    stopper_thread = Thread(target=stopper_fn, args=(stopper_event,))
    main_loop_thread = Thread(
        target=main_loop_fn, args=(cam, fa, stopper_event, state_checker))
    state_checker_thread = Thread(
        target=state_checker_fn, args=(state_checker, stopper_event))
    stopper_thread.start()
    main_loop_thread.start()
    state_checker_thread.start()
