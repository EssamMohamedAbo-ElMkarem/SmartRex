# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:58:05 2020

@author: Essam Mohamed
"""

class Speak(object):
    def __init__(self, text):
        self.engine = pyttsx3.init()
        self.engine.say(text)
        self.engine.runAndWait()

class Recognizer(object):
    response = dict()
    def __init__(self, recognizer, microphone):
        """Transcribe speech from recorded from `microphone`.
        Returns a dictionary with three keys:
        "success": a boolean indicating whether or not the API request was
                   successful
        "error":   `None` if no error occured, otherwise a string containing
                   an error message if the API could not be reached or
                   speech was unrecognizable
        "transcription": `None` if speech could not be transcribed,
                   otherwise a string containing the transcribed text
        """
        # check that recognizer and microphone arguments are appropriate type
        if not isinstance(voice_recognizer, sr.Recognizer):
            raise TypeError("`recognizer` must be `Recognizer` instance")
    
        if not isinstance(microphone, sr.Microphone):
            raise TypeError("`microphone` must be `Microphone` instance")
    
        # adjust the recognizer sensitivity to ambient noise and record audio
        # from the microphone
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            Speak("Talk now")
            audio = recognizer.listen(source)
    
        # set up the response object
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }
    
        # try recognizing the speech in the recording
        # if a RequestError or UnknownValueError exception is caught,
        #     update the response object accordingly
        try:
            response["transcription"] = recognizer.recognize_google(audio)
        except sr.RequestError:
            # API was unreachable or unresponsive
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            # speech was unintelligible
            response["error"] = "Unable to recognize speech"
        self.response = response
        print("You said: " + self.response['transcription'])
    def get_voice(self):
        return self.response
    
    
class FetchKnowledge(object):
    r = Recognizer(voice_recognizer, microphone)
    question = r.get_voice()["transcription"]
    answer = ""
    wolf_id = 'UUPAH6-5RG6P9K9R7'
    def __init__(self, question):
        self.question = question
        self.main_func()
        
    def main_func(self):
        try:
            client = wolframalpha.Client(self.wolf_id) #id from your account on wolframalpha
            res = client.query(self.question)
    
            for pod in res.pods:
                answer = re.sub(r"[^a-zA-Z0-9 :._]", "", str("{p.title} : {p.text}".format(p=pod).encode("UTF-8")))
                self.answer = answer
        except Exception:
            try:
                page = wikipedia.page(self.question)
                self.answer = str(page.content.encode("UTF-8"))
            except Exception:
                answer = "Please try again later and if I were you I would report this to the developer He is not that hard to get"

    def get_answer(self):
        return self.answer
    
        
class FaceRecognizer(object):
    subject_name = ""
    def __init__(self):

        cascade_string = "haarcascade_frontalface_alt2.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_string)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read(r"\trainer.yml")
        self.labels = {}
        with open(r"\labels.pickle", "rb") as f:
            brought_labels = pickle.load(f)
            self.labels = {value: key for key, value in brought_labels.items()}
        self.cap = cv2.VideoCapture(0)

        while True:
            # Capturing frame by frame
            self.ret, self.frame = self.cap.read()
            # Converting from RGB  to Gray scale
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            # Main face detection process
            self.faces = self.face_cascade.detectMultiScale(self.gray, scaleFactor=1.5, minNeighbors=5)
            # Drawing rectangle around faces
            for (x, y, w, h) in self.faces:
                self.roi_gray = self.gray[y:y+h, x:x+w]
                self.roi_color = self.frame[y:y+h, x:x+w]
                # Main Identification process

                self.id_, self.conf = self.recognizer.predict(self.roi_gray)

                if self.conf >= 45:
                    print(self.labels[self.id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    self.subject_name = self.labels[self.id_]
                    cv2.putText(self.frame, self.labels[self.id_], (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

                self.make_rectangle(self.frame, x, y, w, h)
                print(x, y, w, h)

            cv2.imshow("PyFace", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def make_rectangle(frame, x, y, w, h):
        color = (255, 0, 0)  # Stands for Blue in RGB system
        stroke = 2
        end_x = x + w
        end_y = y + h
        cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke)

    @staticmethod
    def train():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, "images")
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []
        
        for root, dirs, files in os.walk(image_dir):
        	for file in files:
        		if file.endswith("png") or file.endswith("jpg"):
        			path = os.path.join(root, file)
        			label = os.path.basename(root).replace(" ", "-").lower()
        			#print(label, path)
        			if not label in label_ids:
        				label_ids[label] = current_id
        				current_id += 1
        			id_ = label_ids[label]
        			#print(label_ids)
        			#y_labels.append(label) # some number
        			#x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
        			pil_image = Image.open(path).convert("L") # grayscale
        			size = (550, 550)
        			final_image = pil_image.resize(size, Image.ANTIALIAS)
        			image_array = np.array(final_image, "uint8")
        			#print(image_array)
        			faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
        
        			for (x,y,w,h) in faces:
        				roi = image_array[y:y+h, x:x+w]
        				x_train.append(roi)
        				y_labels.append(id_)
        
        with open("labels.pickle", 'wb') as f:
        	pickle.dump(label_ids, f)
        
        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trainer.yml")
def main():
    print("         @@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("       *************************** @")
    print("     %%%%%%%%%%%%%%%%%%%%%%%%%%% * @")
    print("   ########################### % * @")
    print("   #   <Welcome to SmartRex> # % * @")
    print("   # ======================= # % *")
    print("   # written by EssamMohamed # %")
    print("   ###########################\n")
    print("1-StartRex :)   2-UpdateRexBase   99-Exit :(")
    command = int(input(">>>"))
    if command == 1:
        #live_vid_thread = threading.Thread(target=VidFeed)
        try:
            app = Recognizer(voice_recognizer, microphone)
            #live_vid_thread.start()
            #live_vid_thread.join()
            print("Executed Successfully!!!")
        except Exception as ex:
            print(ex)
    elif command == 2:
         FaceRecognizer.train()        
    elif command == 99:
        sys.exit()
    else:
        print("Please, Enter a valid command")
        main()
if __name__ == "__main__":
    import speech_recognition as sr
    import numpy as np
    from PIL import Image
    import cv2
    import os
    import sys
    import threading
    import pickle
    import pyttsx3
    import wolframalpha
    import wikipedia
    import re
    voice_recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    main()
    
    
        
