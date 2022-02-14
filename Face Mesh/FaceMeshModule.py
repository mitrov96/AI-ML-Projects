import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, static_mode=False, max_faces=2, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_conf = min_detection_conf
        self. min_tracking_conf = min_tracking_conf
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(self.static_mode, self.max_faces, self.min_detection_conf,
                                                    self.min_tracking_conf)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=1, circle_radius=2)

    def find_face_mesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for face_lms in self.results.multi_face_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, face_lms, self.mp_face_mesh.FACE_CONNECTIONS,
                                                self.draw_spec, self.draw_spec)
                face = []
                for id, lm in enumerate(face_lms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img, faces = detector.find_face_mesh(img)
        if len(faces) != 0:
            print(len(faces))
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
