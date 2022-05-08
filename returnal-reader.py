from curses import is_term_resized
import cv2
import streamlink
import re
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
from PIL import Image
import imutils
import time
from tesserocr import PyTessBaseAPI, PSM
from string import digits


#URL = "https://www.twitch.tv/danger1983__"
URL = "https://youtu.be/c8fK07HwotA"


class CircularBuffer(object):
    def __init__(self, size):
        """initialization"""
        self.index= 0
        self.size= size
        self._data = []

    def record(self, value):
        """append an element"""
        if len(self._data) == self.size:
            self._data[self.index]= value
        else:
            self._data.append(value)
        self.index= (self.index + 1) % self.size

    def __getitem__(self, key):
        """get element by index like a regular array"""
        return(self._data[key])

    def __repr__(self):
        """return string representation"""
        return self._data.__repr__() + ' (' + str(len(self._data))+' items)'

    def get_all(self):
        """return a list of all the elements"""
        return(self._data)

    def all_equal(self):
        return self._data.count(self._data[0]) == len(self._data)  



api = PyTessBaseAPI()
api.SetVariable('tessedit_char_whitelist', 'x0123456789,.')
LABELS = [ "phase", "room", "score", "multi" ]
SIZE = 5


def get_text_out(img, single):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    api.SetPageSegMode(PSM.SINGLE_CHAR if single else PSM.SINGLE_LINE)
    
    api.SetImage(Image.fromarray(gray))
    gray_text = api.GetUTF8Text()
    

    api.SetImage(Image.fromarray(dilation))
    dill_text = api.GetUTF8Text()

    ret = gray_text if gray_text == dill_text else None
    if ret:
        conf = api.AllWordConfidences()[0] 
    else:
        conf = 0
   
    return ret, conf

class Recognizer(object):
    def __init__(self, validator, is_single_digit=False):
        self.last = CircularBuffer(SIZE)
        self.current = 0
        self.validator = validator or (lambda new, old: new)
        self.is_single_digit = is_single_digit
        self.on_new_value_fun = lambda new, old: None
        pass

    def set_on_new_value(self, fun):
        self.on_new_value_fun = fun or (lambda new, old: None)

    def recognize(self, image):
        text, confidence = get_text_out(image, self.is_single_digit)
        if text is None or confidence < 60:
            return

        text = re.sub(r"[^\d]", "", text)
        if len(text.lstrip()) == 0:
            return

        self.last.record(text)

        if not self.last.all_equal():
            return

        value = self.validator(text, self.current)
        if value is None:
            return
        
        if value != self.current:
            self.on_new_value_fun(value, self.current)
        
        self.current = value


def validate_int(new, old):
    try:
        return int(new)
    except ValueError:
        return None


def validate_not_smaller(new, old):
    new = validate_int(new, old)
    return new if new > old else None
    

def validate_phase(new, old):
    new = validate_int(new, old)
    if new == old + 1 or old == 0:
        return new
    return old


def validate_room(new, old):
    new = validate_int(new, old)
    if (
        old == 0 or # initial
        new == old + 1 or # regular progress
        (new == 1 and old == 20) # switch phases
    ): 
        return new
    
    return None


def validate_score(new, old):
    return validate_not_smaller(new, old)
    

def validate_multi(new, old):
    multiplier = validate_int(new, old) 
    if multiplier is None:
        return None

    multiplier = multiplier/100.0
    return multiplier


# MAIN

if __name__ == "__main__":
    streams = streamlink.streams(URL)
    stream_url = streams["best"].to_url()

    cap = cv2.VideoCapture(stream_url)
    fps = FPS().start()

    phase = Recognizer(validate_phase, True)
    room = Recognizer(validate_room)
    score = Recognizer(validate_score)
    multi = Recognizer(validate_multi)

    recognizers = (phase, room, score, multi)
    phase.set_on_new_value(lambda new, old: print(f"Phase => {new}"))
    room.set_on_new_value(lambda new, old: print(f"Room => {new}"))
    score.set_on_new_value(lambda new, old: print(f"Score => {new}"))
    multi.set_on_new_value(lambda new, old: print(f"Multi => {new}"))

    while True:
        ret, frame = cap.read()

        frame = frame[90:426, 1584:1920]

        fphase = frame[33:81, 76:124]
        froom = frame[30:81, 187:252]
        fscore = frame[150:180, 67:307]
        fmulti = frame[223:260, 165:282]

        fragments = (fphase, froom, fscore, fmulti)

        for rec, frag in zip(recognizers, fragments):
            rec.recognize(frag)

        fps.update()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            for label, rec in zip(LABELS, recognizers):
                print(f"{label.capitalize()} = {rec.current}")

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cap.release()
    cv2.destroyAllWindows()
        

