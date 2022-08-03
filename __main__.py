from time import sleep
from asset import AssetHolder
from conf import config
import cv2
import face_alignment
import torch
import numpy as np
import pyvirtualcam
from threading import Event, Thread


# Optionally set detector and some additional detector parameters
face_detector = 'blazeface'  #'sfd'
face_detector_kwargs = {
    # "filter_threshold" : 0.8
}

# Read images to be placed on the output stram
thug = AssetHolder("./assets/thug.png")
ciga = AssetHolder("./assets/ciga.png")

# Run the 3D face alignment on a test image, without CUDA.
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=True,
                                  face_detector=face_detector, face_detector_kwargs=face_detector_kwargs)
cam = cv2.VideoCapture(0)

initial_log_flag = True

stopper_event = Event()


def stopper_fn():
    stopper_event.set()
    # dirty crunch to make console output consistent
    sleep(10)
    while True:
        q = input(">>> input `q` to quit: ")
        print(q)
        if q == 'q':
            print(">>> exiting stopper...")
            stopper_event.clear()
            return


def main_loop_fn():
    with pyvirtualcam.Camera(width=640, height=480, fps=20, backend="v4l2loopback", device="/dev/video2") as vcam:
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
                    eye_bbox = get_eye_bbox(face, frame=None)
                    frame = set_thug(frame, eye_bbox, thug)
                    mouse_width, lower_lip = get_lig_position(face, frame=None)
                    frame = set_ciga(frame, mouse_width, lower_lip, ciga)
            
            global initial_log_flag
            if initial_log_flag:
                print(">>> Stram started...")
                initial_log_flag = False
            
            vcam.send(cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
            vcam.sleep_until_next_frame()

            # cv2.imshow("frame", frame)
            # if cv2.waitKey(config["ms_per_frame"]) & 0xFF == ord("q"):
            #     break
        
        print(">>> exiting main loop...")
        
    cam.release()
    cv2.destroyAllWindows()


def get_eye_bbox(face, frame=None):
    eye_left_x, eye_left_y, eye_right_x, eye_right_y = 0, 0, 0, 0 
    for idx, kp in enumerate(face):
        if idx == 36:
            eye_left_x = kp[0]
            eye_left_y = kp[1]
        if idx == 45:
            eye_right_x = kp[0]
            eye_right_y = kp[1]
    dx = eye_right_x - eye_left_x
    dy = eye_right_y - eye_left_y
    eye_width = np.sqrt(dx**2 + dy**2)

    # get rotation angle
    rot_deg = np.arctan2(dy, dx) * 180 / np.pi
    rot_rad = np.arctan2(dy, dx)
    # print("rot_deg", rot_deg)
    if 85 <= rot_deg <= 95 or -95 <= rot_deg <= -85:
        return np.array([[0, 0],[0, 0],[0, 0],[0, 0]], np.int32)
    
    # set margins
    margin_dy = int(0.60 * dy)
    margin_dx = int(0.60 * dx)
    margin_dh = int(0.25 * eye_width)

    l_x = int(eye_left_x - margin_dx)
    l_y = int(eye_left_y - margin_dy)

    r_x = int(eye_right_x + margin_dx)
    r_y = int(eye_right_y + margin_dy)

    dm_x = np.cos(np.pi/2 + rot_rad) * margin_dh
    dm_y = np.cos(rot_rad) * margin_dh

    tl_x = l_x - dm_x
    tl_y = l_y - dm_y

    bl_x = l_x + dm_x
    bl_y = l_y + dm_y

    tr_x = r_x - dm_x
    tr_y = r_y - dm_y
    
    br_x = r_x + dm_x
    br_y = r_y + dm_y

    bbox = np.array([[int(tl_x), int(tl_y)],[int(tr_x), int(tr_y)],[int(br_x), int(br_y)],[int(bl_x), int(bl_y)]], np.int32)
    return bbox


def set_thug(frame, eye_bbox, thug_asset):
    thug_h = thug_asset.image_rgb.shape[0]
    thug_w = thug_asset.image_rgb.shape[1]

    bbox_thug = np.array([
        [0, 0],           [thug_w, 0], 
        [thug_w, thug_h], [0, thug_h]])
    
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    h, _ = cv2.findHomography(bbox_thug, eye_bbox, cv2.RANSAC, 5.0)

    try:
        alpha_channel_f_warped = cv2.warpPerspective(thug_asset.alpha_channel_f, h, (frame_w, frame_h))
        thug_rgb_f_warped = cv2.warpPerspective(thug_asset.image_rgb_f, h, (frame_w, frame_h))
    except:
        return frame
    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha_channel_f_warped, thug_rgb_f_warped)
    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha_channel_f_warped, frame)
    # Add the masked foreground and background.
    frame = cv2.add(foreground, background)
    return frame


def get_lig_position(face, frame=None):
    upper_lip_x, upper_lip_y, lower_lip_x, lower_lip_y = 0, 0, 0, 0
    left_lip_x, left_lip_y, right_lip_x, right_lip_y = 0, 0, 0, 0
    for idx, kp in enumerate(face):
        if idx == 63:
            upper_lip_x = kp[0]
            upper_lip_y = kp[1]
        if idx == 65:
            lower_lip_x = kp[0]
            lower_lip_y = kp[1]
        if idx == 54:
            left_lip_x = kp[0]
            left_lip_y = kp[1]
        if idx == 48:
            right_lip_x = kp[0]
            right_lip_y = kp[1]
    
    mouse_dx = left_lip_x - right_lip_x
    mouse_dy = left_lip_y - right_lip_y
    mouse_width = np.sqrt(mouse_dx**2 + mouse_dy**2).astype(np.int32)
    lower_lip = np.array([lower_lip_x, lower_lip_y], dtype=np.int32)

    if frame is not None:
        frame = cv2.circle(frame, center=(int(upper_lip_x), int(upper_lip_y)), radius=0, color=(0,0,255), thickness=5)
        frame = cv2.putText(frame, "upper_lip", (int(upper_lip_x), int(upper_lip_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, 2)
        
        frame = cv2.circle(frame, center=(int(lower_lip_x), int(lower_lip_y)), radius=0, color=(255,0,0), thickness=5)
        frame = cv2.putText(frame, "lower_lip", (int(lower_lip_x), int(lower_lip_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, 2)
        
        frame = cv2.circle(frame, center=(int(left_lip_x), int(left_lip_y)), radius=0, color=(255,0,0), thickness=5)
        frame = cv2.putText(frame, "left_lip", (int(left_lip_x), int(left_lip_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, 2)
        
        frame = cv2.circle(frame, center=(int(right_lip_x), int(right_lip_y)), radius=0, color=(255,0,0), thickness=5)
        frame = cv2.putText(frame, "right_lip", (int(right_lip_x), int(right_lip_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, 2)
    
    return mouse_width, lower_lip


def set_ciga(frame, mouse_width, lower_lip, ciga_asset: AssetHolder):

    alpha_channel = ciga_asset.alpha_channel_f.copy()
    image_rgb_f = ciga_asset.image_rgb_f.copy()

    scaler = 1.5
    scale = mouse_width * scaler
    height = scale
    width = image_rgb_f.shape[1] * scale / image_rgb_f.shape[0]

    alpha_channel = cv2.resize(alpha_channel, (int(width), int(height)))
    image_rgb_f = cv2.resize(image_rgb_f, (int(width), int(height)))

    # get roi only after rotation
    roi = frame[
        lower_lip[1]:(lower_lip[1]+alpha_channel.shape[0]), 
        lower_lip[0]:(lower_lip[0]+alpha_channel.shape[1]), 
        :]

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha_channel, image_rgb_f)
    # Multiply the background with ( 1 - alpha )
    if alpha_channel.shape != roi.shape:
        return frame
    background = cv2.multiply(1.0 - alpha_channel, roi)

    # Add the masked foreground and background.
    blended = cv2.add(foreground, background)

    # Put blended image back to the position where it was taken
    frame[
        lower_lip[1]:(lower_lip[1]+alpha_channel.shape[0]), 
        lower_lip[0]:(lower_lip[0]+alpha_channel.shape[1]), 
        :] = blended
    
    return frame


if __name__ == "__main__":
    stopper_thread = Thread(target=stopper_fn)
    main_loop_thread = Thread(target=main_loop_fn)
    stopper_thread.start()
    main_loop_thread.start()
