import cv2
import subprocess as sp
from cv2 import resize
import numpy as np
from imutils.video import FPS


class ReturnalVideoCapture():
    def __init__(self, url):
        self.url = url
        self.fps = FPS().start()
    
    def read(self):
        self.fps.update()
        return False, None

    def get_frame_no(self):
        return self.fps._numFrames;

class ReturnalCV2Capture(ReturnalVideoCapture):
    def __init__(self, url):
        super().__init__(url)
        self.capture = cv2.VideoCapture(url)
    
    def read(self):
        super().read()
        ret, frame = self.capture.read()
        return ret, frame

    def release(self):
        self.capture.release()

    

        

class ReturnalFFMPEGCapture(ReturnalVideoCapture):
    def __init__(self, url, width, height):
        super().__init__(url)
        self.width = width
        self.height = height
        self.frame_size = 3 * width * height
        self.pipe = sp.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'quiet', '-i', url, '-f', 'image2pipe', '-pix_fmt', 'bgr24', '-vcodec', 'rawvideo', '-'],
                             stdin=sp.PIPE, stdout=sp.PIPE, bufsize=self.frame_size)
    
    def read(self):
        super().read()
        raw = self.pipe.stdout.read(self.frame_size)
        frame = np.fromstring(raw, dtype='uint8').reshape(self.height, self.width, 3)
        return True, frame        

    def release(self):
        self.pipe.terminate()

