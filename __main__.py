from time import sleep
from asset_handler import ThugAsset, CigaAsset
import cv2
import pyvirtualcam
from threading import Event, Lock, Thread
from config import vcamera_conf, face_mesh_conf
from utils import StateChecker
import mediapipe as mp

# Read images to be placed on the output stram
thug = ThugAsset("./assets/thug.png")
ciga = CigaAsset("./assets/ciga.png")


def stopper_fn(stopper_event: Event) -> None:
    stopper_event.set()
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


def main_loop_fn(camera: cv2.VideoCapture,
                 stopper_event: Event,
                 state_checker: StateChecker) -> None:
    initial_log_flag = True
    enabler = state_checker.content
    mp_face_mesh = mp.solutions.face_mesh

    with pyvirtualcam.Camera(**vcamera_conf) as vcam,\
        mp_face_mesh.FaceMesh(**face_mesh_conf) as face_mesh:

        print(f'>>> Using virtual camera: {vcam.device}')
        while stopper_event.is_set() and camera.isOpened():
            # get hot reload state
            if state_checker.is_content_available():
                enabler = state_checker.get_actual_content()
                state_checker.disable_content_access()
                print("\nconfig changed\n>>>")

            success, image = camera.read()
            if not success:
                continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            faces = None
            if enabler["enable_dnn"]:
                faces = face_mesh.process(image)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if faces and faces.multi_face_landmarks:
                for landmark_list in faces.multi_face_landmarks:
                    if enabler["enable_thug"]:
                        image = thug.apply_asset(image, landmark_list)
                    if enabler["enable_ciga"]:
                        image = ciga.apply_asset(image, landmark_list)
                    if initial_log_flag:
                        print(">>> Stream started...")
                        initial_log_flag = False
                    vcam.send(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    vcam.sleep_until_next_frame()
            else:
                vcam.send(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                vcam.sleep_until_next_frame()

    camera.release()
    cv2.destroyAllWindows()
    print(">>> exiting main loop...")


if __name__ == "__main__":

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
        target=main_loop_fn, args=(cam, stopper_event, state_checker))
    state_checker_thread = Thread(
        target=state_checker_fn, args=(state_checker, stopper_event))
    stopper_thread.start()
    main_loop_thread.start()
    state_checker_thread.start()
