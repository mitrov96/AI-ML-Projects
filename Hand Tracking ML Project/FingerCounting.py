import cv2
import time
import os
import HandTrackingModule as htm

width_cam = 1260
height_cam = 780

cap = cv2.VideoCapture(0)
cap.set(3, width_cam)
cap.set(4, height_cam)

folder_path = 'FingersImages'
list = sorted(os.listdir(folder_path))

previous_time = 0
overlay_list = []

for img_path in list:
    image = cv2.imread(f'{folder_path}/{img_path}')
    overlay_list.append(image)

detector = htm.HandDetector(detectionCon=0.75)

tip_ids = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    lm_list = detector.findPosition(img, draw=False)
    if len(lm_list) != 0:
        fingers = []
        # Thumb
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Other 4 fingers
        for id in range(1, 5):
            if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)
        print(total_fingers)

        h, w, c = overlay_list[total_fingers-1].shape
        img[0:h, 0:w] = overlay_list[total_fingers-1]

        cv2.rectangle(img, (20, 225), (170, 425), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, str(total_fingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (0, 0, 0), 25)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(img, f'FPS: {int(fps)}', (1150, 650), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
