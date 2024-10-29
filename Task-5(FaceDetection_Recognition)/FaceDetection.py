import cv2
import os
import tkinter as tk
from tkinter import messagebox
import face_recognition
import pickle

dataset = "frames"
encodings_file = "encodings.pkl"
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

known_faces = {}

if not os.path.exists(dataset):
    os.makedirs(dataset)

def precompute_encodings():
    global known_faces
    known_faces = {}  

    for person in os.listdir(dataset):
        person_dir = os.path.join(dataset, person)
        encodings = []

        for img_file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_file)
            person_image = face_recognition.load_image_file(img_path)

            encodings_in_image = face_recognition.face_encodings(person_image)
            if len(encodings_in_image) > 0:
                encodings.append(encodings_in_image[0])  

        if encodings:
            known_faces[person] = encodings  

    with open(encodings_file, "wb") as f:
        pickle.dump(known_faces, f)
    print("Encodings precomputed and stored.")

def load_encodings():
    global known_faces
    if os.path.exists(encodings_file):
        with open(encodings_file, "rb") as f:
            known_faces = pickle.load(f)
        print("Encodings loaded from file.")

def register():
    name = input("Enter your name: ")
    path = os.path.join(dataset, name)

    if not os.path.isdir(path):
        os.mkdir(path)
        print(f"Created directory for {name}")

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Cannot open camera")
        return

    initial_frames = 0
    total_frames = 30
    encodings = []  
    while initial_frames < total_frames:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        face_locations = face_recognition.face_locations(rgb_frame)  # Get face locations

        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            face_frame = rgb_frame[top:bottom, left:right]

            face_encodings = face_recognition.face_encodings(rgb_frame, [face_location])
            if len(face_encodings) > 0:
                encodings.append(face_encodings[0])  # Store the encoding

            frame_filename = os.path.join(path, f'frame_{initial_frames}.jpg')
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame: {frame_filename}")
            initial_frames += 1

        cv2.imshow('Register - Live Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if encodings:
        known_faces[name] = encodings

        with open(encodings_file, "wb") as f:
            pickle.dump(known_faces, f)
        print(f"Encodings for {name} saved.")

    display_login_and_exit()


def recognize():
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Cannot open camera")
        return

    recognized = False
    max_attempts = 10  
    attempt_count = 0

    while not recognized and attempt_count < max_attempts:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            for person, person_encodings in known_faces.items():
                matches = face_recognition.compare_faces(person_encodings, face_encoding)

                if True in matches:
                    print(f"Face recognized: {person}")
                    messagebox.showinfo("Login Success", f"Welcome, {person}!")
                    recognized = True
                    break

        attempt_count += 1  
        cv2.imshow('Login - Face Recognition', frame)

        if recognized:
            break

    cam.release()
    cv2.destroyAllWindows()

    if not recognized:
        messagebox.showerror("Login Failed", "Face not recognized!")

    root.deiconify()

def display_login_and_exit():
    for widget in root.winfo_children():
        widget.destroy()  

    login_button = tk.Button(root, text="Login", width=20, command=start_recognition)
    login_button.pack(pady=10)

    exit_button = tk.Button(root, text="Exit", width=20, command=root.destroy)
    exit_button.pack(pady=10)

def start_registration():
    root.withdraw()  
    register()
    root.deiconify()  

def start_recognition():
    root.withdraw()  
    recognize()

def create_gui():
    global root
    root = tk.Tk()
    root.title("Face Recognition System")

    register_button = tk.Button(root, text="Register", width=20, command=start_registration)
    register_button.pack(pady=10)

    login_button = tk.Button(root, text="Login", width=20, command=start_recognition)
    login_button.pack(pady=10)

    root.mainloop()

load_encodings()

create_gui()
