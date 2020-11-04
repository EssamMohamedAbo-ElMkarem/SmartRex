# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 21:58:05 2020

@author: Essam Mohamed
"""
import speech_recognition as sr
import numpy as np
from PIL import Image
import cv2
import os
import sys
import shutil
import threading
import webbrowser
import pickle
import pyttsx3
import wolframalpha
import wikipedia
import re
voice_recognizer = sr.Recognizer()
input_microphone = sr.Microphone()
current_profile_name = "World"
answer_ = "Hello World Rex is Ready"
spoken_times = 0


class Profile(object):
    name = ""
    age = ""
    profession = ""
    favourite_artists = []
    favourite_music = []

    def __init__(self, name, age, favourite_artists, favourite_music, profession):
        self.name = name
        self.age = age
        self.profession = profession
        self.favourite_artist = favourite_artists
        self.favourite_music = favourite_music

    def create_profile(self):
        with open("images/" + self.name + "/" + self.name + ".rex", "w") as data:
            data.write(self.name)
            data.write(self.age)
            data.write(self.profession)
            data.writelines(self.favourite_artists)
            data.writelines(self.favourite_music)


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
        "error":   `None` if no error occurred, otherwise a string containing
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

    def get_voice(self):
        return self.response


class FetchKnowledge(object):
    question: str = ""
    answer: str = ""
    wolf_id = 'UUPAH6-5RG6P9K9R7'

    def __init__(self, question):
        self.question = question
        self.main_func()

    def main_func(self):
        try:
            client = wolframalpha.Client(self.wolf_id)  # id from your account on wolframalpha
            res = client.query(self.question)
            self.answer = next(res.results).text

        except Exception as ex:
            print(ex)
            try:
                page = wikipedia.page(self.question)
                self.answer = str(page.content)
            except Exception as ex:
                print(ex)
                answer = "Please try again later and if I were you I would report this to the developer" \
                         " He is not that hard to get"
                self.answer = answer

    def get_answer(self):
        return self.answer


class FaceRecognizer(object):

    def __init__(self):
        global current_profile_name
        cascade_string = "haarcascade_frontalface_alt2.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_string)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer.yml")
        self.labels = {}
        with open("labels.pickle", "rb") as f:
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
                self.roi_gray = self.gray[y:y + h, x:x + w]
                self.roi_color = self.frame[y:y + h, x:x + w]
                # Main Identification process

                self.id_, self.conf = self.recognizer.predict(self.roi_gray)

                if self.conf >= 75:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    current_profile_name = self.labels[self.id_]
                    cv2.putText(self.frame, self.labels[self.id_], (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                self.make_rectangle(self.frame, x, y, w, h)

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
        print("Training Started...")
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
                    # print(label, path)
                    if label not in label_ids:
                        label_ids[label] = current_id
                        current_id += 1
                    id_ = label_ids[label]
                    # print(label_ids)
                    # y_labels.append(label) # some number
                    # x_train.append(path) # verify this image, turn into a NUMPY array, GRAY
                    pil_image = Image.open(path).convert("L")  # grayscale
                    size = (550, 550)
                    final_image = pil_image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(final_image, "uint8")
                    # print(image_array)
                    faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                    for (x, y, w, h) in faces:
                        roi = image_array[y:y + h, x:x + w]
                        x_train.append(roi)
                        y_labels.append(id_)

        with open("labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)

        recognizer.train(x_train, np.array(y_labels))
        recognizer.save("trainer.yml")
        print("Training finished Successfully!!")
        main()


def toi(txt="", word=""):
    index = txt.find(word)
    return txt[index + len(word) - 1:]


def rec_thread():
    global answer_, spoken_times
    while True:
        rec = Recognizer(voice_recognizer, input_microphone)
        txt = rec.get_voice()["transcription"]
        txt = txt.lower()
        if txt is None:
            pass
        else:
            print("You said " + txt)
            if txt.find("shut down") == -1:
                if txt.find("question") == -1:
                    if txt == "who am i":
                        answer_ = "You are " + current_profile_name
                    elif txt == "who is your creator":
                        answer_ = "Essam Mohamed is my creator"
                    elif txt.find("youtube") >= 0:
                        vid = toi(txt, "youtube")
                        webbrowser.open("https://www.youtube.com/results?search_query=" + vid)
                    elif txt.find("open website") >= 0:
                        website = toi(txt, "open website")
                        webbrowser.open(website)
                else:
                    question = toi(txt, "question")
                    print("Your question is " + question)
                    data = FetchKnowledge(question=question)
                    ans = data.get_answer()
                    answer_ = ans
                    spoken_times = 0
                print(answer_)
            else:
                os._exit(0)


def speak_thread():
    global spoken_times
    while True:
        if spoken_times == 0:
            Speak(answer_)
            spoken_times = spoken_times + 1
        else:
            pass


def take_list(taken_list=None, list_name=""):
    if taken_list is None:
        taken_list = []
    while True:
        new_fav = input("Enter the " + list_name + " to add to your list of favourites or 99 to finish>>> ")
        if new_fav == '99':
            break
        else:
            taken_list.append(new_fav)


def main():
    print("         @@@@@@@@@@@@@@@@@@@@@@@@@@@")
    print("       *************************** @")
    print("     %%%%%%%%%%%%%%%%%%%%%%%%%%% * @")
    print("   ########################### % * @")
    print("   #   <Welcome to SmartRex> # % * @")
    print("   # ======================= # % *")
    print("   # written by EssamMohamed # %")
    print("   ###########################\n")
    print("1-StartRex :)   2-UpdateRexBase   3-AddNewProfile   4-RemoveProfile   5-UpdateImageDB   99-Exit :(")
    command = int(input(">>>"))
    if command == 1:
        vid_feed_thread = threading.Thread(target=FaceRecognizer)
        voice_rec_thread = threading.Thread(target=rec_thread)
        rex_voice_thread = threading.Thread(target=speak_thread)
        try:
            vid_feed_thread.start()
            voice_rec_thread.start()
            rex_voice_thread.start()
            vid_feed_thread.join()
            voice_rec_thread.join()
            rex_voice_thread.join()
            print("Executed Successfully!!!")
        except Exception as ex:
            print(ex)
    elif command == 2:
        os.remove("labels.pickle")
        os.remove("trainer.yml")
        FaceRecognizer.train()
    elif command == 3:
        new_name = input("Enter your name>>> ")
        new_age = input("Enter your age>>> ")
        new_profession = input("Enter your profession>>> ")
        favourite_artists = []
        favourite_music = []
        take_list(favourite_artists, 'artist')
        take_list(favourite_music, 'type of music')

        path = "images/" + new_name + "/"
        if os.path.isdir(path):
            print("This user already exists try another username the next time!")
            main()
        else:
            os.mkdir(path)
            new_profile = Profile(new_name, new_age, new_profession, favourite_artists, favourite_music)
            new_profile.create_profile()
            cap = cv2.VideoCapture(0)
            counter = 1
            while True:
                _, frame = cap.read()
                cv2.imshow("NewProfile", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('s'):
                    cv2.imwrite(path + str(counter) + ".jpg", frame)
                    print("saved successfully!")
                    counter += 1
                elif k == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    elif command == 4:
        user_to_remove = input("Enter the username you want to remove>>> ")
        path = "images/" + user_to_remove + "/"
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            print("The username you entered doesn't exist try a valid username the next time!")
            main()
    elif command == 5:
        existing_name = input("Enter  username to update >>> ")
        path = "images/" + existing_name + "/"
        if os.path.isdir(path):
            counter = len(os.listdir(path)) + 1
            cap = cv2.VideoCapture(0)
            while True:
                _, frame = cap.read()
                cv2.imshow("NewProfile", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('s'):
                    cv2.imwrite(path + str(counter) + ".jpg", frame)
                    print("saved successfully!")
                    counter += 1
                elif k == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print("This user doesn't exist try another username the next time!")
            main()

    elif command == 99:
        os._exit(0)
    else:
        print("Please, Enter a valid command")
        main()


if __name__ == "__main__":
    main()
