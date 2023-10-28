from djitellopy import tello
import KeyPressedModule as kp
import numpy as np
from threading import Thread
import time
import cv2
import math

######## PARAMETERS ###########

fSpeed = 585 / 50  # Forward Speed in cm/s   (15cm/s)
aSpeed = 360 / 10  # Angular Speed Degrees/s  (50d/s)
interval = 0.25
dInterval = fSpeed * interval
aInterval = aSpeed * interval

###############################################

x, y = 500, 500
a = 0
yaw = 0
kp.init()
me = tello.Tello()
me.connect()
me.streamon()

print(me.get_battery())
points = [(0, 0), (0, 0)]
me.send_rc_control(0, 0, 25, 0)

# Variables for capturing images and recording videos
image_counter = 0
video_counter = 0
video_writer = None


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

    if kp.getKey("q"): me.land(); time.sleep(3)
    if kp.getKey("e"):
        me.takeoff()

    time.sleep(interval)
    a += yaw
    x += int(d * math.cos(math.radians(a)))
    y += int(d * math.sin(math.radians(a)))

    return [lr, fb, ud, yv, x, y]


# Function to draw points on the image
def drawPoints(img, points):
    for point in points:
        cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, points[-1], 8, (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'({(points[-1][0] - 500) / 100},{(points[-1][1] - 500) / 100})m',
                (points[-1][0] + 10, points[-1][1] + 30), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 0, 255), 1)


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
points = [(500, 500)]  # Initial point
while True:
    # Drawing and displaying points
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    img = np.zeros((1000, 1000, 3), np.uint8)
    if points[-1][0] != vals[4] or points[-1][1] != vals[5]:
        points.append((vals[4], vals[5]))

    drawPoints(img, points)
    cv2.imshow("Points Plotting", img)

    # Capture a frame from the video stream
    frame = me.get_frame_read().frame

    # Convert frame to RGB format for displaying true color image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Tello Video Stream", rgb_frame)

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

    # Integration for recording videos
    if key == ord('r') and not keepRecording:
        keepRecording = True
        recorder.start()
    elif key == ord('t') and keepRecording:
        keepRecording = False

    # Press 'q' to exit the loop and close the window
    if key == ord('q'):
        keepRecording = False  # Stop recording if 'q' is pressed
        break

cv2.destroyAllWindows()
me.streamoff()  # Turn off video streaming
me.end()
