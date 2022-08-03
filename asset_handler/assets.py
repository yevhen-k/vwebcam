from typing import Tuple
import cv2
import numpy as np
import abc


__all__ = [
    'ThugAsset',
    'CigaAsset',
]


class AssetHolder(abc.ABC):

    def __init__(self, asset_path: str) -> None:
        image = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        assert image.shape[2] == 4, f"Image {asset_path} doesn't have transparancy layer"
        self.alpha_channel = image[:, :, 3]
        self.image_rgb = image[:, :, :3]
        self.alpha_channel = cv2.cvtColor(
            self.alpha_channel, cv2.COLOR_GRAY2BGR)
        self.alpha_channel_f = self.alpha_channel.astype(float) / 255
        self.image_rgb_f = self.image_rgb.astype(float) / 255

    @abc.abstractmethod
    def apply_asset(frame: np.array, face: np.array) -> np.array:
        ...


class ThugAsset(AssetHolder):

    left_eye_kp_idx = 36
    right_eye_kp_idx = 45

    margin_dy_scale = 0.60
    margin_dx_scale = 0.60
    margin_dh_scale = 0.25

    def __init__(self, asset_path: str) -> None:
        super().__init__(asset_path)

    def apply_asset(self, frame: np.array, face: np.array) -> np.array:
        eye_bbox = self.__get_eye_bbox(face)
        frame = self.__set_thug(frame, eye_bbox)
        return frame

    def __get_eye_bbox(self, face: np.array) -> np.array:
        eye_left_x, eye_left_y, eye_right_x, eye_right_y = 0, 0, 0, 0
        for idx, kp in enumerate(face):
            if idx == self.left_eye_kp_idx:
                eye_left_x = kp[0]
                eye_left_y = kp[1]
            if idx == self.right_eye_kp_idx:
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
            return np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.int32)

        # set margins

        margin_dy = int(self.margin_dy_scale * dy)
        margin_dx = int(self.margin_dx_scale * dx)
        margin_dh = int(self.margin_dh_scale * eye_width)

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

        bbox = np.array([[int(tl_x), int(tl_y)], [int(tr_x), int(tr_y)], [
                        int(br_x), int(br_y)], [int(bl_x), int(bl_y)]], np.int32)
        return bbox

    def __set_thug(self, frame: np.array, eye_bbox: np.ndarray) -> np.array:
        thug_h = self.image_rgb.shape[0]
        thug_w = self.image_rgb.shape[1]

        bbox_thug = np.array([
            [0, 0],           [thug_w, 0],
            [thug_w, thug_h], [0, thug_h]])

        frame_h = frame.shape[0]
        frame_w = frame.shape[1]

        h, _ = cv2.findHomography(bbox_thug, eye_bbox, cv2.RANSAC, 5.0)

        try:
            alpha_channel_f_warped = cv2.warpPerspective(
                self.alpha_channel_f, h, (frame_w, frame_h))
            thug_rgb_f_warped = cv2.warpPerspective(
                self.image_rgb_f, h, (frame_w, frame_h))
        except:
            return frame

        # Multiply the foreground with the alpha matte
        foreground = cv2.multiply(alpha_channel_f_warped, thug_rgb_f_warped)
        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha_channel_f_warped, frame)
        # Add the masked foreground and background.
        frame = cv2.add(foreground, background)
        return frame


class CigaAsset(AssetHolder):

    upper_lip_idx = 63
    lower_lip_idx = 65
    left_lip_idx = 54
    right_lip_idx = 48

    scaler = 1.5

    def __init__(self, asset_path: str) -> None:
        super().__init__(asset_path)

    def apply_asset(self, frame: np.array, face: np.array) -> np.array:
        mouse_width, lower_lip = self.__get_lip_position(face)
        frame = self.__set_ciga(frame, mouse_width, lower_lip)
        return frame

    def __get_lip_position(self, face: np.array) -> Tuple[int, np.array]:

        upper_lip_x, upper_lip_y, lower_lip_x, lower_lip_y = 0, 0, 0, 0
        left_lip_x, left_lip_y, right_lip_x, right_lip_y = 0, 0, 0, 0

        for idx, kp in enumerate(face):
            if idx == self.upper_lip_idx:
                upper_lip_x = kp[0]
                upper_lip_y = kp[1]
            if idx == self.lower_lip_idx:
                lower_lip_x = kp[0]
                lower_lip_y = kp[1]
            if idx == self.left_lip_idx:
                left_lip_x = kp[0]
                left_lip_y = kp[1]
            if idx == self.right_lip_idx:
                right_lip_x = kp[0]
                right_lip_y = kp[1]

        mouse_dx = left_lip_x - right_lip_x
        mouse_dy = left_lip_y - right_lip_y
        mouse_width = np.sqrt(mouse_dx**2 + mouse_dy**2).astype(np.int32)
        lower_lip = np.array([lower_lip_x, lower_lip_y], dtype=np.int32)

        return mouse_width, lower_lip

    def __set_ciga(self, frame: np.array, mouse_width: int, lower_lip: np.array) -> np.array:

        alpha_channel = self.alpha_channel_f.copy()
        image_rgb_f = self.image_rgb_f.copy()

        scale = mouse_width * self.scaler
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
