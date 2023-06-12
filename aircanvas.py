import numpy as np
import cv2
from collections import deque
from hand_tracking_module import HandDetector
from detection import CharacterDetector
from time import sleep


tipIds = [4, 8, 12, 16, 20]

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
vpoints = [deque(maxlen=1024)]

black_index = 0
green_index = 0
red_index = 0
voilet_index = 0

kernel = np.ones((5, 5), np.uint8)
colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
colorIndex = 0

cv2.namedWindow("Paint", cv2.WINDOW_AUTOSIZE)

cap = cv2.VideoCapture(0)

while not cap.isOpened():
    sleep(0.5)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 150)

paint_window_width = int(cap.get(3))
paint_window_height = int(cap.get(4))
paintWindow = np.zeros((paint_window_height, paint_window_width, 3)) + 0xFF
detector = HandDetector(detectionCon=0.75)
det = CharacterDetector(loadFile="recognizer.h5")


def draw_line(frame, pt1, pt2, color):
    cv2.line(frame, pt1, pt2, color, 20)

while True:
    (_, frame) = cap.read(0)

    frame = cv2.flip(frame, 1)
    img = detector.findHands(frame)
    lmList = detector.findPosition(img, draw=False)

    fingers = []

    if len(lmList) != 0:
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # print(lmList[4])
        totalFingers = fingers.count(1)

    frame = cv2.circle(frame, (40, 90), 20, (255, 255, 255), -1)
    frame = cv2.circle(frame, (40, 140), 20, (0, 0, 0), -1)
    frame = cv2.circle(frame, (40, 190), 20, (255, 0, 0), -1)
    frame = cv2.circle(frame, (40, 240), 20, (0, 255, 0), -1)
    frame = cv2.circle(frame, (40, 290), 20, (0, 0, 255), -1)
    frame = cv2.rectangle(frame, (520, 1), (630, 65), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "C",
        (32, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Recognise",
        (530, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        2,
        cv2.LINE_AA,
    )

    center = None
    # If the contours are formed

    if len(lmList) != 0 and totalFingers == 1:
        # Get the radius of the enclosing circle
        # around the found contour
        lst = lmList[tipIds[fingers.index(1)]]
        x, y = lst[1], lst[2]

        # Draw the circle around the contour

        cv2.circle(frame, (x, y), int(20), (0, 0xFF, 0xFF), 2)

        # Calculating the center of the detected contour
        center = (x, y)

        # Now checking if the user wants to click on
        # any button above the screen

        if center[0] <= 60:
            # Clear Button

            if 70 <= center[1] <= 110:
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                vpoints = [deque(maxlen=512)]

                black_index = 0
                green_index = 0
                red_index = 0
                voilet_index = 0

                paintWindow[:, :, :] = 0xFF
            elif 120 <= center[1] <= 160:
                colorIndex = 0  # Black
            elif 170 <= center[1] <= 210:
                colorIndex = 1  # Voilet
            elif 220 <= center[1] <= 260:
                colorIndex = 2  # Green
            elif 270 <= center[1] <= 310:
                colorIndex = 3  # Red
        elif 520 < center[0] < 630 and 1 < center[1] < 65:
            print(det.predict(np.float32(paintWindow)))

        else:
            if colorIndex == 0:
                bpoints[black_index].appendleft(center)
            elif colorIndex == 1:
                vpoints[voilet_index].appendleft(center)
            elif colorIndex == 2:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 3:
                rpoints[red_index].appendleft(center)
    else:
        # Append the next deques when nothing is
        # detected to avois messing up

        bpoints.append(deque(maxlen=512))
        black_index += 1
        vpoints.append(deque(maxlen=512))
        voilet_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1

    # Draw lines of all the colors on the
    # canvas and frame

    points = [bpoints, vpoints, gpoints, rpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue

                draw_line(
                    frame, points[i][j][k - 1], points[i][j][k], colors[i]
                )
                draw_line(
                    paintWindow, points[i][j][k - 1], points[i][j][k], colors[i]
                )

    cv2.imshow('Tracking', frame)
    cv2.imshow("Paint", paintWindow)

    # If the 'q' key is pressed then stop the application
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("c"):
        # Clear the canvas
        bpoints = [deque(maxlen=512)]
        gpoints = [deque(maxlen=512)]
        rpoints = [deque(maxlen=512)]
        vpoints = [deque(maxlen=512)]

        black_index = 0
        green_index = 0
        red_index = 0
        voilet_index = 0

        paintWindow = np.zeros((paint_window_height, paint_window_width, 3)) + 0xFF

    if key == ord("r"):
        # Predict the letter
        print(det.predict(np.float32(paintWindow)))

    if key == ord("1"):
        colorIndex = 0

    if key == ord("2"):
        colorIndex = 1

    if key == ord("3"):
        colorIndex = 2

    if key == ord("4"):
        colorIndex = 3

cap.release()
cv2.destroyAllWindows()
