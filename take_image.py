import datetime
import cv2
class TakeImage():
    def take_image(self):
        camera_port = 0
        ramp_frames = 30
        camera = cv2.VideoCapture(camera_port)
        def get_image():
            retval, im = camera.read()
            return im
        for i in range(ramp_frames):
            temp = get_image()
        print("Taking image...")
        camera_capture = get_image()
        currentDT = datetime.datetime.now()
        currentDTSTR = str(currentDT).replace(' ', '')
        file = "images/{}image.png".format(currentDTSTR)
        cv2.imwrite(file, camera_capture)
        del(camera)

if __name__ == '__main__':
    im = TakeImage()
    im.take_image()
