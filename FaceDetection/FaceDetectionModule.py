import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, min_detection_con=0.5):
        self.min_detection_con = min_detection_con
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_con)

    def find_faces(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(img_rgb)
        bboxs = []
        if results.detections:
            for id, detection in enumerate(results.detections):
                bbox_c = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih), int(bbox_c.width * iw), int(bbox_c.height * ih)
                bboxs.append([id, bbox, detection.score])
                img = self.better_draw(img, bbox)
                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        return img, bboxs

    def better_draw(self, img, bbox, l=30, t=5):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 255, 255), 2)
        cv2.line(img, (x, y), (x + l, y), (0, 255, 255), t)
        cv2.line(img, (x, y), (x, y + l), (0, 255, 255), t)

        # Top Right corner
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 255, 255), t)

        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 255), t)

        # Bottom Right corner
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 255), t)

        return img


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceDetector()
    while True:
        success, img = cap.read()
        img, bboxs = detector.find_faces(img)
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
