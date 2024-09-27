#from django.shortcuts import render , HttpResponse
import string
import speech_recognition as sr
# import os

import cv2
import numpy as np
import pickle
import mediapipe as mp
from django.http import StreamingHttpResponse
from django.shortcuts import render
from googletrans import Translator


labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'I_Love_You', 5: "Thank_You", 6: "Please", 7: "Hi", 8: "F", 9: "Good", 10: "No",12:"D",14:"G",15:"H",16:"I",17:"K",18:"L",21:"P",24:"T",25:"U",26:"V",27:"W",28:"X",29:"Y",31:"D",32:"L",33:"U",34:"V",35:"A",36:"O",37:"Yes",38:"S",39:"Fun",42:"Child",44:"Hello",45:"Food/Eat",46:"Cigarette",47:"Baby",48:"Thirsty",49:"You",50:"More"}

model = None
mp_drawing = None
mp_drawing_styles=None
hands = None
mp_hands = None

def load_model_and_hands():
    global model, hands ,mp_drawing_styles ,mp_drawing ,mp_hands
    # Load the model
    model_dict = pickle.load(open(r"D:\Projects\ML\Sign Language\Sign__Language_main\Sign__Language_main\Sign Language\signsign\user\model.p", 'rb'))
    model = model_dict['model']
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3, max_num_hands=2)

def generate_frames():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                x_ = []
                y_ = []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min(x_))
                    data_aux.append(landmark.y - min(y_))

            if len(results.multi_hand_landmarks) == 1:
                data_aux.extend([0] * (42 * 2))

            try:
                prediction = model.predict([np.asarray(data_aux[:84])])
                predicted_character = labels_dict[int(prediction[0])]
                cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3, cv2.LINE_AA)
            except Exception as e:
                print(f"Error during prediction: {e}")

        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    cap.release()

def video_feed(request):
    #print("Hello--v")
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def func():
    isl_gif = ['hello','goodmorning','again','agree','award','celebrate','done','fun','good','great','hello','hurry','morning','name','no','what',"what'sup",'when','where','which','who','why','yes']
    arr = list(string.ascii_lowercase)
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("I am Listening")
        audio = r.listen(source)
        try:
            a = r.recognize_google(audio)
            a = a.lower()
            print('You Said:', a)
            for c in string.punctuation:
                a = a.replace(c, "")
            p=a.split()
            images = []
            for a in p:
                if a in isl_gif:
                    gif_path = f'ISL_Gifs/{a}.gif'
                    images.append(gif_path)
                else:
                    for i in a:
                        if i in arr:
                            image_path = f'letters/{i}.jpg'
                            images.append(image_path)
            context = {'images':images,'txt':a}
            return context
        except Exception as e:
            print("Error:", e)
            return None

def fun(text):
    tra=Translator()
    #text=tra.translate(text,dest='ta').text
    isl_gif = ['hello','goodmorning','again','agree','award','celebrate','done','fun','good','great','hello','hurry','morning','name','no','what',"what'sup",'when','where','which','who','why','yes']
    arr = list(string.ascii_lowercase)
    a=text
    a = a.lower()
    print('You Said:', a)
    for c in string.punctuation:
        a = a.replace(c, "")
    p=a.split()
    print(p)
    images = []
    for a in p:
        if a in isl_gif:
            gif_path = f'ISL_Gifs/{a}.gif'
            images.append(gif_path)
        else:
            for i in a:
                if i in arr:
                    image_path = f'letters/{i}.jpg'
                    images.append(image_path)
    context = {'images':images,'txt':text}
    return context

# Create your views here.
def hello(request):
    if(request.method=='POST'):
        print("Got here")
        result = func()
        return render(request, 'home.html', result)
    else:
        return  render(request,'home.html')

def tex(request):
    if(request.method=='POST'):
        result = fun(request.POST.get('text'))
        return render(request, 'home.html', result)
    else:
        return  render(request,'home.html')
    
def index(request):
    load_model_and_hands()
    #print("Hello World")
    return render(request, 'index.html')
