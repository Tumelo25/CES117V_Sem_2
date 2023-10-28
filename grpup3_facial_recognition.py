from djitellopy import Tello
import KeyPressedModule as kp
import face_recognition
import numpy as np
from threading import Thread
import time
import os
import cv2
import math

######## PARAMETERS ###########

fSpeed = 585 / 50  # Forward Speed in cm/s   (15cm/s)
aSpeed = 360 / 10  # Angular Speed Degrees/s  (50d/s)
interval = 0.25
dInterval: float = fSpeed * interval
aInterval = aSpeed * interval

# Initialize Tello drone and other variables
x, y = 500, 500
a = 0
yaw = 0
kp.init()
me = Tello()
me.connect()
me.streamon()
me.send_rc_control(0, 0, 25, 0)  # Adjust the initial command based on your requirement
print(me.get_battery())

# Variables for capturing images and recording videos
image_counter = 0
video_counter = 0
video_writer = None

# Load known faces and their corresponding encodings
known_face_encodings = []
known_face_names = []

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load known faces from the "known_faces" directory
known_faces_dir = "Images/known_faces"
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        face_image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))

        # Check if a face is detected in the image
        face_locations = face_recognition.face_locations(face_image)
        if len(face_locations) > 0:
            # If a face is detected, get its encodings
            face_encoding = face_recognition.face_encodings(face_image)[0]

            # Rest of your code to use the face_encoding
            known_face_encodings.append(face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No face detected in the image: {filename}")


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    aspeed = 50

    global x, y, yaw, a
    d = 0

    if kp.getKey("LEFT"):
        lr = -speed
        d = dInterval
        a = -180

    elif kp.getKey("RIGHT"):
        lr = speed
        d = -dInterval
        a = 180

    if kp.getKey("UP"):
        fb = speed
        d = dInterval
        a = 270

    elif kp.getKey("DOWN"):
        fb = -speed
        d = -dInterval
        a = -90

    if kp.getKey("w"):
        ud = speed

    elif kp.getKey("s"):
        ud = -speed

    if kp.getKey("a"):
        yv = -aspeed
        yaw -= aInterval

    elif kp.getKey("d"):
        yv = aspeed
        yaw += aInterval

    if kp.getKey("q"):
        me.land()
        time.sleep(3)

    if kp.getKey("e"):
        me.takeoff()

    time.sleep(interval)
    a += yaw
    x += int(d * math.cos(math.radians(a)))
    y += int(d * math.sin(math.radians(a)))

    return [lr, fb, ud, yv, x, y]


# Function to calculate PID control signal
def pid_control(target, current):
    Kp = 0.5  # Proportional gain
    Ki = 0.5  # Integral gain
    Kd = 0.1  # Derivative gain
    prev_error = 0
    integral = 0

    error = target - current
    integral += error
    derivative = error - prev_error
    output = Kp * error + Ki * integral + Kd * derivative

    prev_error = error
    return int(output)


# Function to calculate PID control signal to track the face
def track_face(frame, face_location):
    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2

    # Calculate the center of the detected face
    face_center_x = (face_location[0] + face_location[2]) // 2
    face_center_y = (face_location[1] + face_location[3]) // 2

    # Calculate errors in x (horizontal) and y (vertical) directions
    error_x = frame_center_x - face_center_x
    error_y = frame_center_y - face_center_y

    # Use PID control to adjust the drone's left/right movement based on the error in x direction
    pid_output_x = pid_control(0, error_x)

    # Use PID control to adjust the drone's up/down movement based on the error in y direction
    pid_output_y = pid_control(0, error_y)

    # Return the PID control outputs for both x and y directions
    return pid_output_x, pid_output_y


# Function to handle video recording
def videoRecorder():
    global video_counter
    height, width, _ = frame.shape
    video_filename = f'Videos/Recorded_video_{int(time.time())}+{video_counter}.mp4'
    video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
    print(f"Recording started. Video will be saved as '{video_filename}'!")

    while keepRecording:
        # Convert frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_writer.write(rgb_frame)
        time.sleep(1 / 30)
    video_writer.release()
    video_counter += 1
    print('Recording stopped!')


keepRecording = False  # Variable to control video recording
recorder = Thread(target=videoRecorder)

# Main loop
while True:
    # Keyboard input for manual control
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    # Capture a frame from the video stream
    frame = me.get_frame_read().frame

    # Convert frame to RGB format for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert into grayscale
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around the detected faces on the frame
    for (x, y, w, h) in faces:
        face_location = (x, y, x + w, y + h)

        # Get the face encoding for the current face
        face_encoding = face_recognition.face_encodings(rgb_frame, [(y, x + w, y + h, x)])[0]

        # Compare the current face encoding with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match is found, use the name of the known face
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a rectangle around the detected face and display the name
        if name != "Unknown":
            color = (0, 255, 0)  # Green color for known faces
        else:
            color = (0, 0, 255)  # Red color for unknown faces

        # Draw a rectangle around the detected face and display the name
        cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), color, 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(rgb_frame, name, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

        # Calculate PID control signals for left/right and up/down movements based on face position
        pid_output_x, pid_output_y = track_face(rgb_frame, face_location)

        # Convert pid_output_x and pid_output_y to int type
        pid_output_x = int(pid_output_x)
        pid_output_y = int(pid_output_y)

        # Send the PID control signals to adjust left/right and up/down movements
        me.send_rc_control(pid_output_x, pid_output_y, 0, 0)  # Adjust other control signals as needed

    # Check for key presses using cv2.waitKey()
    key = cv2.waitKey(1) & 0xFF

    # Integration for capturing images
    if key == ord('c'):  # Check if 'c' key is pressed
        # Convert frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save the frame as an image
        image_filename = f'Images/captured_image_{int(time.time())}+{image_counter}.jpg'
        cv2.imwrite(image_filename, rgb_frame)
        print(f"Picture captured and saved as '{image_filename}'!")
        image_counter += 1
    # Resize the frame to the desired dimensions (360x240)
    resized_frame = cv2.resize(rgb_frame, (360, 400))

    # Display the frame with face recognition and streaming
    cv2.imshow("Tello Video Stream", resized_frame)

    # Integration for recording videos
    if key == ord('r') and not keepRecording:
        keepRecording = True
        recorder.start()
    elif key == ord('t') and keepRecording:
        keepRecording = False

    # Press 'q' to exit the loop and close the window
    if key == ord('q'):
        break

# Release resources and stop video streaming
cv2.destroyAllWindows()
me.streamoff()
me.end()
