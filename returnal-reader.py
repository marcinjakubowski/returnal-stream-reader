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
    if multiplier is None or multiplier > 10000:
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
        self.is_correct = False
        self._last_debug = ""
        self.disable_validation = False
        self.last_known_previous = -1
        self.count_invalid = 0

    def trigger_validation(self, is_enabled):
        if not is_enabled:
            self.last_known_previous = self.current if self.current != 0 else self.last_known_previous
            self.current = 0
        else:
            self.current = self.last_known_previous
            self.last_known_previous = -1

    def set_on_new_value(self, fun):
        self.on_new_value_fun = fun or (lambda new, old: None)

    def recognize(self, image):
        self.is_correct, self.is_new = self._recognize(image)
        return self.is_correct

    def _debug(self, message):
            if self.debug and self._last_debug != message:
                self._last_debug = message
                logging.debug(f"{self.debug} > {message}")

    def _recognize(self, image):
        text, confidence = get_text_out(image, self.is_single_digit, self.debug)
        if text is None or confidence < 60:
            return False, False

        text = re.sub(r"[^\d]", "", text)
        if len(text.lstrip()) == 0:
            self._debug(f"Empty text")
            return False, False

        self.last.record(text)
        self.last_correction.record(text)

        if not self.last.all_equal():
            self._debug(f"Not equal {text}")
            return False, False
        
        value = self.validator(text, self.current)
        if value is None:
            self.count_invalid += 1
            self._debug(f"Not valid {text} vs {self.current}")
            # check correction
            if self.count_invalid > 10 and self.last_correction.all_equal():
                self._debug(f"Correcting from {self.current} to {text}")
                value = self.validator(text, self.previous)

            if value is None:
                return False, False
        
        self.count_invalid = 0

        if value != self.current:
            self.on_new_value_fun(value, self.current)
            self.previous = self.current
            self.current = value
            if self.debug:
                logging.info(f"{self.debug} > New value = {value}")
            return True, True
        
        return True, False

class ReturnalRecognizer():
    def __init__(self, id, capture, logger=logging):
        self.id = id
        self.capture = capture
        self.logger = logger
        self.title = None
        self.room = Recognizer(validate_room)
        def validate_phase(new, old):
            new = validate_int(new, old)
            if (new == old + 1 or old == 0) and (self.room.current in (-1, 0, 1)):
                return new
            return old
        self.phase = Recognizer(validate_phase, True)
        self.score = Recognizer(validate_score)
        self.multi = Recognizer(validate_multi)
        self.phase.set_on_new_value(self.get_logger("Phase"))
        self.score.set_on_new_value(self.get_logger("Score"))
        self.multi.set_on_new_value(self.get_logger("Multi"))
        self.room.set_on_new_value(self.get_logger("Room"))
        self.recognizers = (self.phase, self.room, self.score, self.multi)
        self.last_frames = {
            self.phase: -1000,
            self.room: -1000,
            self.score: -1000,
            self.multi: -1000
        }
        self.frameno = -1
        self.wait_for = [None]
        self.wait_for_skip = []
        self.wait_for_timeout = -1
        self.wait_for_rerecognize = 1
        self.skipping = 0
        self.skip = 0
        self.skipped_new_room = False

        self.crop_w, self.crop_h = self._get_frame_crop()

    def get_logger(self, logger_name):
        def _logger(new, old):
            t = self.capture.get_time()
            m = int((t / 1000) / 60)
            s = int(t / 1000) % 60
            self.logger.info(f"{m:02d}:{s:02d} | {logger_name} => {new}")
        return _logger



    def _is_recent(self, rec):
        return self.last_frames[rec] >= self.frameno - 30

    def _read_capture(self):
        if self.skipping > 0:
            self.capture.skip(self.skipping)
        _, frame = self.capture.read()
        self.frameno = self.capture.get_frame_no()
        return frame


    def update(self):
        original = self._read_capture()
        frame = original[self.crop_h, self.crop_w]
        frame = cv2.resize(frame, (336, 336))
        fragments = frame[30:81, 70:144], frame[30:81, 187:252], frame[150:180, 67:307], frame[223:260, 165:282]
        
        for rec, frag in zip(self.recognizers, fragments):
            if rec.recognize(frag):
                self.last_frames[rec] = self.frameno

        do_further_recognition = self._recognize_pause_restart()
        do_further_recognition = self._recognize_skipped(do_further_recognition)
        do_further_recognition = self._recognize_new_room(do_further_recognition)
        

        return original, frame

    def _recognize_skipped(self, should_recognize):
        if not should_recognize or self.wait_for_rerecognize != -1:
            self.wait_for_skip = []
            return False
        
        if self.wait_for_timeout != -1 or self.skip == 0:
            return True

        if self.skipping == self.skip and not self.wait_for_skip:
            self.wait_for_skip = [*self.recognizers]
            self.skipping = 0
        elif self.skipping == 0 and self.skip > 0 and not self.wait_for_skip:
            self.skipping = self.skip
            self.skipped_new_room = False
            logging.debug(f"Skipping for {self.skip} frames")

        if self.wait_for_skip:
            self.wait_for_skip = list(filter(lambda rec: not rec.is_correct, self.wait_for_skip))
            if self.phase.is_new or self.room.is_new:
                self.skipped_new_room = True

            if not self.wait_for_skip:
                if self.skipped_new_room:
                    self.wait_for_timeout = 0
                return self.skipped_new_room

        return False            
    
    def _recognize_pause_restart(self):
        # check whether something new has been recorded in a while
        last_frame_recorded_anything = max(self.last_frames.values())
        is_anything_recent = last_frame_recorded_anything > self.frameno - 300

        if self.wait_for_rerecognize == -1 and not is_anything_recent:
             logging.debug("Waiting for new recognition")
             self.wait_for_rerecognize = 1
             [rec.trigger_validation(False) for rec in self.recognizers]
             return False
        elif self.wait_for_rerecognize > 0 and is_anything_recent:
            if (self.phase.last_known_previous == -1 and 
                self.room.last_known_previous == -1 and
                self.score.last_known_previous == -1 and
                self.multi.last_known_previous == -1
                ):
                self.logger.debug(" REC | New video detected")
                DB.start_new_run(self.id)
                self.wait_for_rerecognize = -1
            elif ((self.score.is_new and self.score.current == self.score.last_known_previous)
                  or (self.multi.is_new and self.multi.current == self.multi.last_known_previous)):
                self.logger.debug(" REC | Previous run detected")
                [rec.trigger_validation(True) for rec in self.recognizers]
                self.phase.is_new = False
                self.room.is_new = False
                self.score.is_new = False
                self.multi.is_new = False
                self.wait_for_rerecognize = -1
            elif (   (self.phase.is_correct and self.phase.current == 1)
                 and (self.room.is_correct and self.room.current == 1)):
                self.logger.debug(" REC | Recognized new run")
                self.multi.current = 1
                self.score.current = 1
                self.wait_for_rerecognize = -1
                DB.start_new_run(self.id)
            
        
        return True

    def _recognize_new_room(self, should_recognize):
        if not should_recognize:
            return False

        # new room that hasn't been recorded
        if (self.phase.is_new or self.room.is_new) and self.wait_for_timeout == -1:
            self.wait_for_timeout = 100
            self.wait_for = [rec for rec in [self.multi, self.room, self.phase] if not self._is_recent(rec)]
            if self.room.current == 1 and self.phase not in self.wait_for:
                self.wait_for.append(self.phase)
        
        elif self.wait_for_timeout >= 0:
            self.wait_for = list(filter(lambda rec: not rec.is_correct, self.wait_for))
            self.wait_for_timeout = self.wait_for_timeout - 1 if self.wait_for_timeout > 0 else self.wait_for_timeout

        still_wait = not self.wait_for or (self.wait_for_timeout == 0 and self.multi.current > 0)
        if (self.phase.current > 0 and self.room.current > 0) and still_wait:
            if self.room.current == 1 and self.phase in self.wait_for:
                self.phase.current += 1
            time = self.capture.get_time()
            DB.record(time, self.phase.current, self.room.current, self.score.current, self.multi.current)
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

DEFAULT="returnal.mp4"

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%H:%M:%S', level=logging.DEBUG)

    if len(sys.argv) == 1:
        sys.argv.append(DEFAULT)

    if sys.argv[1].startswith("http"):
        logging.info("Using FFMPEG")
        stream_url = get_stream_url(sys.argv[1])
        cap = ReturnalFFMPEGCapture(stream_url, 1920, 1080)
    else:
        logging.info("Using OpenCV")
        cap = ReturnalCV2Capture(sys.argv[1])

    recognize = ReturnalRecognizer(sys.argv[1], cap, logging.getLogger(sys.argv[1]))
    #recognize.skip = 50
    title = cv2.resize(recognize.title, (640, 480))
    paused = False
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("z"):
            for label, rec in zip(LABELS, recognize.recognizers):
                print(f"{label} = {rec.current}")
        elif key == ord("r"):
            DB.start_new_run(sys.arvgv[1])
            recognize = ReturnalRecognizer(sys.argv[1], cap)

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
        

