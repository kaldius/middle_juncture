import cv2
import dlib
import face_recognition
from threading import Thread


class Camera:
    def __init__(self):
        self.data = None

        # specify video capture api as v4l2
        self.cam = cv2.VideoCapture(0, cv2.CAP_V4L2)

        self.WIDTH = 1280
        self.HEIGHT = 720

        # in order to get 30fps, need to set codec as motion-JPG
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cam.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cam.set(cv2.CAP_PROP_FPS, 30)

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

        self.ZOOM_SPEED = 1
        self.zoom_scale = 1
        self.zoom_goal = 1

        self.TRANSLATION_SPEED = 1
        self.center_goal = [int(self.WIDTH / 2), int(self.HEIGHT / 2)]
        self.center_current = [int(self.WIDTH / 2), int(self.HEIGHT / 2)]

        # fmt: skip
        self.LIFETIME = 100 # no. of frames where we track a face before searching for faces again
        self.trim_vert = 0.35 # top  |----a----+--b--| bottom         ratio of a/(a+b)
        self.trim_horiz = 0.5 # left |----c----+--d--| right          ratio of c/(c+d)
        self.trim_zoom = 0.5 # face_height/frame_height
        self.downtime = 30 # when no faces detected, no. of frames to wait before searching for faces again
        self.current_track = None # current tracking object, format: [tracking, no_of_frames_left], where no_of_frames_left starts at LIFETIME
        

    def zoom(self, img):
        height, width = img.shape[:2]
        z_height, z_width = int(height / self.zoom_scale), int(width / self.zoom_scale)

        self.zoom_scale += (self.zoom_goal - self.zoom_scale) / 10 * self.ZOOM_SPEED

        min_y = int(self.center_current[1] - (z_height / 2))
        max_y = int(self.center_current[1] + (z_height / 2))

        min_x = int(self.center_current[0] - (z_width / 2))
        max_x = int(self.center_current[0] + (z_width / 2))

        if min_y < 0:
            max_y -= min_y
            min_y = 0
        elif max_y > height:
            min_y -= max_y - height
            max_y = height

        if min_x < 0:
            max_x -= min_x
            min_x = 0
        elif max_x > width:
            min_x -= max_x - width
            max_x = width
        # print(f"min_x: {min_x}\nmax_x: {max_x}\nmin_y: {min_y}\nmax_y: {max_y}\n")

        cropped = img[min_y:max_y, min_x:max_x]

        return cv2.resize(cropped, (width, height))

    def update_center(self):
        # print("center_current: " + str(self.center_current))
        dx, dy = (
            self.center_goal[0] - self.center_current[0],
            self.center_goal[1] - self.center_current[1],
        )
        self.center_current[0] += int(dx / 10) * self.TRANSLATION_SPEED
        self.center_current[1] += int(dy / 10) * self.TRANSLATION_SPEED

    def find_and_track_faces(self, image):
        faces = face_recognition.face_locations(image, model="cnn")
        if len(faces) == 0:
            self.current_track = None
            return
        face = faces[0]
        face = dlib.rectangle(face[3], face[0], face[1], face[2])
        tracker = dlib.correlation_tracker()
        tracker.start_track(image, face)
        self.current_track = [tracker, self.LIFETIME]

    def zoom_rect(self, rect):
        face_height = rect.bottom() - rect.top()
        # to reduce jitters, check if new center_goal and zoom_goal are close to the original

        new_center_goal = [
            int((rect.right() - rect.left()) * self.trim_horiz + rect.left()),
            int(face_height * self.trim_vert + rect.top()),
        ]
        if self.euclidean_distance(self.center_goal, new_center_goal) > 30:
            self.center_goal = new_center_goal

        new_zoom_goal = self.HEIGHT / face_height * self.trim_zoom
        # print("curent zoom: " + str(self.zoom_goal) + "new zoom: " + str(new_zoom_goal))
        if abs(self.zoom_goal - new_zoom_goal) > 0.5:
            self.zoom_goal = new_zoom_goal
        # self.zoom_goal = 1 / self.trim_zoom

    def euclidean_distance(self, a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    # optional
    def show_face_rect(self, image, pos):
        pos = dlib.rectangle(
            int(pos.left()),
            int(pos.top()),
            int(pos.right()),
            int(pos.bottom()),
        )
        cv2.rectangle(
            image, (pos.left(), pos.top()), (pos.right(), pos.bottom()), (100, 200, 100)
        )

    def stream(self):
        def streaming():
            self.ret = True
            while self.ret:
                self.ret, np_image = self.cam.read()
                if np_image is None:
                    continue
                if self.current_track is None:
                    if self.downtime > 30:
                        # print("finding faces")
                        self.downtime = 0
                        self.find_and_track_faces(np_image)
                    else:
                        self.downtime += 1
                else:
                    tracker, age = self.current_track
                    if age <= 0 or tracker.update(np_image) <= 7:
                        # print("reset")
                        self.current_track = None
                        self.find_and_track_faces(np_image)
                    else:
                        pos = tracker.get_position()
                        self.zoom_rect(pos)
                        # self.show_face_rect(np_image, pos)
                        self.current_track[1] -= 1
                self.update_center()
                np_image = self.zoom(np_image)
                self.data = np_image
                k = cv2.waitKey(1)
                if k == ord("q"):
                    self.release()
                    break

        Thread(target=streaming).start()

    def zoom_in(self):
        if self.trim_zoom < 1:
            self.trim_zoom += 0.1
        # print("zoom goal: " + str(self.zoom_goal))

    def zoom_out(self):
        if self.trim_zoom > 0:
            self.trim_zoom -= 0.1
        # print("zoom goal: " + str(self.zoom_goal))

    def up(self):
        self.trim_vert -= 0.1
        # print("vert: " + str(self.trim_vert))

    def down(self):
        self.trim_vert += 0.1
        # print("vert: " + str(self.trim_vert))

    def left(self):
        self.trim_horiz -= 0.1
        # print("horiz: " + str(self.trim_horiz))

    def right(self):
        self.trim_horiz += 0.1
        # print("horiz: " + str(self.trim_horiz))

    def show(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        while True:
            frame = self.data
            if frame is not None:
                cv2.imshow("middlejuncture", frame)

            key = cv2.waitKey(1)

            if key == ord("q"):
                self.release()
                break
            elif key == ord("z"):
                self.zoom_in()
            elif key == ord("x"):
                self.zoom_out()
            elif key == ord("w"):
                self.up()
            elif key == ord("s"):
                self.down()
            elif key == ord("a"):
                self.left()
            elif key == ord("d"):
                self.right()

    def release(self):
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cam = Camera()
    cam.stream()
    cam.show()
