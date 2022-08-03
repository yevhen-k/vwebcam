import cv2

class AssetHolder:
    def __init__(self, asset_path: str) -> None:
        image = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        assert image.shape[2] == 4, f"Image {asset_path} doesn't have transparancy layer"
        self.alpha_channel = image[:,:,3]
        self.image_rgb = image[:,:,:3]
        self.alpha_channel = cv2.cvtColor(self.alpha_channel, cv2.COLOR_GRAY2BGR)
        self.alpha_channel_f = self.alpha_channel.astype(float) / 255
        self.image_rgb_f = self.image_rgb.astype(float) / 255
