import sys
import cv2
import streamlink
import re
from circular_buffer import CircularBuffer
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, PSM
from db import ReturnalDb
from video_capture import ReturnalFFMPEGCapture

api = PyTessBaseAPI()
api.SetVariable('tessedit_char_whitelist', 'x0123456789,.')
LABELS = [ "phase", "room", "score", "multi" ]
SIZE = 10


def get_stream_url(url):
    streams = streamlink.streams(url)
    stream = streams['best']
    if isinstance(stream, streamlink.stream.http.HTTPStream):
        return stream.to_url()
    else:
        return stream.substreams[0].to_url()
    

def get_text_out(img, single, debug):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, (127,), (255,))
    if np.count_nonzero(mask) / np.size(mask) <= 0.03:
        return None, 0
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    api.SetPageSegMode(PSM.SINGLE_CHAR if single else PSM.SINGLE_LINE)
    
    api.SetImage(Image.fromarray(thresh))
    text = api.GetUTF8Text()

    if debug is not None:
        cv2.imshow(debug, thresh)
        
    

    ret = text
    if ret:
        conf = api.AllWordConfidences()[0] 
    else:
        conf = 0
   
    return ret, conf

class Recognizer(object):
    def __init__(self, validator, is_single_digit=False):
        self.debug = None
        self.last = CircularBuffer(SIZE)
        self.current = 0
        self.validator = validator or (lambda new, old: new)
        self.is_single_digit = is_single_digit
        self.on_new_value_fun = lambda new, old: None
        pass

    def set_on_new_value(self, fun):
        self.on_new_value_fun = fun or (lambda new, old: None)

    def recognize(self, image):
        text, confidence = get_text_out(image, self.is_single_digit, self.debug)
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


DB = ReturnalDb()
SHOULD_RECORD = -1

def on_new_room(new, old):
    global SHOULD_RECORD
    SHOULD_RECORD = 20
    print(f"Room => {new}")    

# MAIN
def get_recognizers():
    phase = Recognizer(validate_phase, True)
    
    room = Recognizer(validate_room)
    score = Recognizer(validate_score)
    multi = Recognizer(validate_multi)  
    multi.debug = "Multi"



    phase.set_on_new_value(lambda new, old: print(f"Phase => {new}"))
    score.set_on_new_value(lambda new, old: print(f"Score => {new}"))
    multi.set_on_new_value(lambda new, old: print(f"Multi => {new}"))
    room.set_on_new_value(on_new_room)

    return (phase, room, score, multi)

def get_frame_crop(capture):
    _, frame = capture.read()

    w1 = 1584
    w2 = 1920
    h1 = 90
    h2 = 426
    
    wr = 1
    hr = 1

    if frame.shape[0] != 1080:
        wr = frame.shape[1] / 1920
        hr = frame.shape[0] / 1080
        w1 = int(w1 * wr)
        w2 = int(w2 * wr)
        h1 = int(h1 * hr)
        h2 = int(h2 * hr)
    
    return slice(w1, w2), slice(h1, h2)



if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.append("https://youtu.be/TrrxdfSFJJw")

    if sys.argv[1].startswith("http"):
        stream_url = get_stream_url(sys.argv[1])
        cap = ReturnalFFMPEGCapture(stream_url, 1920, 1080)
    else:
        cap = cv2.VideoCapture(sys.argv[1])
        cap.set(cv2.CAP_PROP_POS_MSEC, 1180000)

    recognizers = get_recognizers()

    w, h = get_frame_crop(cap)
    while True:
        _, frame = cap.read()
        frame = frame[h, w]
        frame = cv2.resize(frame, (336, 336))
        fphase = frame[25:57, 76:124]
        froom = frame[30:81, 187:252]
        fscore = frame[150:180, 67:307]
        fmulti = frame[223:260, 165:282]

        fragments = (fphase, froom, fscore, fmulti)
        for rec, frag in zip(recognizers, fragments):
            rec.recognize(frag)

        if SHOULD_RECORD > 0:
            SHOULD_RECORD -= 1

        if SHOULD_RECORD == 0:
            SHOULD_RECORD = -1
            phase, room, score, multi = recognizers
            DB.record(phase.current, room.current, score.current, multi.current)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("s"):
            for label, rec in zip(LABELS, recognizers):
                print(f"{label.capitalize()} = {rec.current}")
        elif key == ord("r"):
            DB.start_new_run()
            recognizers = get_recognizers()

    # stop the timer and display FPS information
    cap.fps.stop()
    print("[INFO] elasped time: {:.2f}".format(cap.fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(cap.fps.fps()))

    cap.release()
    cv2.destroyAllWindows()
        

