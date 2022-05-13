import sys
import cv2
import logging
import streamlink
import re
from circular_buffer import CircularBuffer
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, PSM
from db import ReturnalDb
from video_capture import ReturnalCV2Capture, ReturnalFFMPEGCapture

api = PyTessBaseAPI()
api.SetVariable('tessedit_char_whitelist', 'x0123456789,.')
LABELS = [ "Phase", "Room", "Score", "Multi" ]
CONFIDENCE_SIZE = 7
DB = ReturnalDb()

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
    
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    api.SetPageSegMode(PSM.SINGLE_CHAR if single else PSM.SINGLE_LINE)
    
    api.SetImage(Image.fromarray(thresh))
    text = api.GetUTF8Text()

    if debug is not None:
        cv2.imshow(debug, thresh)
        
    if np.count_nonzero(mask) / np.size(mask) <= 0.03:
        return None, 0

    ret = text
    if ret:
        conf = api.AllWordConfidences()[0] 
    else:
        conf = 0
   
    return ret, conf

def validate_int(new, old):
    try:
        return int(new)
    except ValueError:
        return None


def validate_not_smaller(new, old):
    new = validate_int(new, old)
    return new if new >= old else None


def validate_room(new, old):
    new = validate_int(new, old)
    if 1 <= new <= 20 and (
        old == 0 or # initial
        new == old + 1 or # regular progress
        new == old or 
        (new == 1 and old == 20) # switch phases

    ): 
        return new
    
    return None


def validate_score(new, old):
    new = validate_not_smaller(new, old)
    if new is None:
        return new
    if old <= 10 or new < 10000:
        return new
    if new <= old + 15000:
        return new
    if new <= old * 1.5:
        return new

    return None

def validate_multi(new, old):
    multiplier = validate_int(new, old) 
    if multiplier is None:
        return None

    multiplier = multiplier/100.0
    return multiplier    

class Recognizer(object):
    def __init__(self, validator, is_single_digit=False):
        self.debug = None
        self.last = CircularBuffer(CONFIDENCE_SIZE)
        self.last_correction = CircularBuffer(CONFIDENCE_SIZE * 3)
        self.current = 0
        self.previous = 0
        self.validator = validator or (lambda new, old: new)
        self.is_single_digit = is_single_digit
        self.on_new_value_fun = lambda new, old: None
        self.is_new = False
        self._last_debug = ""
        self.disable_validation = False
        self.last_known_previous = -1
        self.count_invalid = 0

    def trigger_validation(self, is_enabled):
        if not is_enabled:
            self.last_known_previous = self.current
            self.current = 0
        else:
            self.current = self.last_known_previous
            self.last_known_previous = -1

    def set_on_new_value(self, fun):
        self.on_new_value_fun = fun or (lambda new, old: None)

    def recognize(self, image):
        self.is_new = self._recognize(image)
        return self.is_new

    def _debug(self, message):
            if self.debug and self._last_debug != message:
                self._last_debug = message
                logging.debug(f"{self.debug} > {message}")

    def _recognize(self, image):
        text, confidence = get_text_out(image, self.is_single_digit, self.debug)
        if text is None or confidence < 60:
            return False

        text = re.sub(r"[^\d]", "", text)
        if len(text.lstrip()) == 0:
            self._debug(f"Empty text")
            return False

        self.last.record(text)
        self.last_correction.record(text)

        if not self.last.all_equal():
            self._debug(f"Not equal {text}")
            return False
        
        value = self.validator(text, self.current)
        if value is None:
            self.count_invalid += 1
            self._debug(f"Not valid {text} vs {self.current}")
            # check correction
            if self.count_invalid > 10 and self.last_correction.all_equal():
                self._debug(f"Correcting from {self.current} to {text}")
                value = self.validator(text, self.previous)

            if value is None:
                return False
        
        self.count_invalid = 0

        if value != self.current:
            self.on_new_value_fun(value, self.current)
            self.previous = self.current
            self.current = value
            if self.debug:
                logging.info(f"{self.debug} > New value = {value}")
            return True
        
        return False


class ReturnalRecognizer():
    def __init__(self, capture):
        self.capture = capture
        self.title = None
        self.room = Recognizer(validate_room)
        self.room.set_on_new_value(lambda new, old: logging.info(f"Room => {new}"))
        def validate_phase(new, old):
            new = validate_int(new, old)
            if (new == old + 1 or old == 0) and (self.room.current in range(1, 20)):
                return new
            return old
        self.phase = Recognizer(validate_phase, True)
        self.phase.set_on_new_value(lambda new, old: logging.info(f"Phase => {new}"))
        self.score = Recognizer(validate_score)
        self.score.set_on_new_value(lambda new, old: logging.info(f"Score => {new}"))
        self.multi = Recognizer(validate_multi)
        self.multi.set_on_new_value(lambda new, old: logging.info(f"Multi => {new}"))
        self.recognizers = (self.phase, self.room, self.score, self.multi)
        self.last_frames = {
            self.phase: -1,
            self.room: -1,
            self.score: -1,
            self.multi: -1
        }
        self.frameno = -1
        self.wait_for = [None]
        self.wait_for_timeout = -1
        self.wait_for_rerecognize = -1

        self.crop_w, self.crop_h = self._get_frame_crop()

    def _is_recent(self, rec):
        return self.last_frames[rec] >= self.frameno - 30

    def _read_capture(self):
        _, frame = cap.read()
        self.frameno = cap.get_frame_no()
        return frame


    def update(self):
        original = self._read_capture()
        frame = original[self.crop_h, self.crop_w]
        frame = cv2.resize(frame, (336, 336))
        fragments = frame[25:57, 76:124], frame[30:81, 187:252], frame[150:180, 67:307], frame[223:260, 165:282]
        
        for rec, frag in zip(self.recognizers, fragments):
            if rec.recognize(frag):
                self.last_frames[rec] = self.frameno

        do_further_recognition = self._recognize_pause_restart()
        self._recognize_new_room(do_further_recognition)

        return original, frame
    
    def _recognize_pause_restart(self):
        # check whether something new has been recorded in a while
        last_frame_recorded_anything = max(self.last_frames.values())
        is_anything_recent = last_frame_recorded_anything > self.frameno - 500

        if self.wait_for_rerecognize == -1 and not is_anything_recent:
             logging.debug("Waiting for new recognition")
             self.wait_for_rerecognize = 1
             [rec.trigger_validation(False) for rec in self.recognizers]
             return False
        elif self.wait_for_rerecognize > 0 and is_anything_recent:
            if (   (self.phase.is_new and self.phase.current == 1)
                 or (self.room.is_new and self.room.current == 1)):
                logging.debug("Recognized new run")
                self.wait_for_rerecognize = -1
                DB.start_new_run()
            elif (   (self.phase.is_new and self.phase.current == self.phase.last_known_previous)
                  or (self.room.is_new and self.room.current == self.room.last_known_previous)
                  or (self.score.is_new and self.score.current == self.score.last_known_previous)
                  or (self.multi.is_new and self.multi.current == self.multi.last_known_previous)):
                logging.debug("Previous run detected")
                [rec.trigger_validation(True) for rec in self.recognizers]
                self.phase.is_new = False
                self.room.is_new = False
                self.score.is_new = False
                self.multi.is_new = False
                self.wait_for_rerecognize = -1
        
        return True

    def _recognize_new_room(self, should_recognize):
        if not should_recognize:
            return False
        
        # new room that hasn't been recorded
        if (self.phase.is_new or self.room.is_new) and self.wait_for_timeout == -1:
            self.wait_for_timeout = 100
            self.wait_for = [rec for rec in [self.multi, self.room, self.phase] if not self._is_recent(rec)]
        if self.wait_for_timeout > 0:
            self.wait_for = list(filter(lambda rec: not rec.is_new, self.wait_for))
            self.wait_for_timeout -= 1
        if (self.phase.current > 0 and self.room.current > 0) and (not self.wait_for or self.wait_for_timeout == 0):
            DB.record(self.phase.current, self.room.current, self.score.current, self.multi.current)
            self.wait_for_timeout = -1
            self.wait_for = [None]

        return True

    def _get_frame_crop(self):
        _, frame = self.capture.read()

        w1 = 1584
        w2 = 1920
        h1 = 90
        h2 = 426
        
        ratio_width = 1
        ratio_height = 1

        if frame.shape[0] != 1080:
            ratio_width = frame.shape[1] / 1920
            ratio_height = frame.shape[0] / 1080
            w1 = int(w1 * ratio_width)
            w2 = int(w2 * ratio_width)
            h1 = int(h1 * ratio_height)
            h2 = int(h2 * ratio_height)
        self.title = frame
        
        return slice(w1, w2), slice(h1, h2)

DEFAULT="returnal.webm"

if __name__ == "__main__":
    logging.basicConfig(format='[%(levelname)s @ %(asctime)s] %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    if len(sys.argv) == 1:
        sys.argv.append(DEFAULT)

    if sys.argv[1].startswith("http"):
        logging.info("Using FFMPEG")
        stream_url = get_stream_url(sys.argv[1])
        cap = ReturnalFFMPEGCapture(stream_url, 1920, 1080)
    else:
        logging.info("Using OpenCV")
        cap = ReturnalCV2Capture(sys.argv[1])

    recognize = ReturnalRecognizer(cap)
    title = cv2.resize(recognize.title, (640, 480))
    paused = False
    skipping = False
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("s"):
            skipping = not skipping
        elif key == ord("z"):
            for label, rec in zip(LABELS, recognize.recognizers):
                print(f"{label} = {rec.current}")
        elif key == ord("r"):
            DB.start_new_run()
            recognize = ReturnalRecognizer(cap)

        if paused:
            continue
            
        frame, subframe = recognize.update()
        display = cv2.resize(frame, (800, 450))
        cv2.imshow("Current", display)

    # stop the timer and display FPS information
    cap.fps.stop()
    print("[INFO] elasped time: {:.2f}".format(cap.fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(cap.fps.fps()))

    cap.release()
    cv2.destroyAllWindows()
        

