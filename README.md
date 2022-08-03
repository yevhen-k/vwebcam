## Prerequesties
1. Install linux kernel headers for your kernel, for example, `linux510-headers`
2. Install `v4l2loopback-dkms`
3. Initialize loopback device as described in [docs](https://github.com/umlaeute/v4l2loopback#options):
   ```bash
   sudo modprobe v4l2loopback devices=1 card_label="My Fake Webcam" exclusive_caps=1
   ls -l /dev/video*
   ```
4. In my case `/dev/video2` was initialized
5. Star the app
6. Check if it workd
   ```bash
   ffplay /dev/video2
   ```
   Window with web stream should appear.
7. After you're done:
   ```bash  
   sudo modprobe --remove v4l2loopback
   ```

## TODOs
- [ ] massive refactoring
- [ ] scale down input images to speed-up calculations (may make keypoint prediction more noisy)
- [x] add margins to make better look and feel for thug mask
- [x] add rotations to the thug mask
- [x] add thug cigarette
- [ ] ability to add new images via config / gui
- [ ] hot reload: config update should automatically trigger update stream
- [x] implement virtual camera support
- [ ] use face [mash](https://google.github.io/mediapipe/solutions/face_mesh.html)

## References
1. PyImageSearch: https://pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
2. Face Alignment: https://github.com/1adrianb/face-alignment
3. PyVirtualCam: https://github.com/letmaik/pyvirtualcam
4. How to project an image in perspective view of a background image — OpenCV, Python. Anshul Sachdev: https://medium.com/acmvit/how-to-project-an-image-in-perspective-view-of-a-background-image-opencv-python-d101bdf966bc
5. https://stackoverflow.com/questions/17822585/copy-blend-images-of-different-sizes-using-opencv
6. https://stackoverflow.com/questions/56002672/display-an-image-over-another-image-at-a-particular-co-ordinates-in-opencv
7. https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html
8. Create fake web cam stream on linux: https://www.linuxfordevices.com/tutorials/linux/fake-webcam-streams
9. Installing linux headers for v4l2loopback-dkms: https://forum.manjaro.org/t/2021-11-19-stable-update-error-in-install-dkms-modules-step/91107
10. Inter Thread communication in Python: https://dotnettutorials.net/lesson/inter-thread-communication-in-python/