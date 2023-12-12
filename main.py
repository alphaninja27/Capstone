# import cv2
# import mediapipe as mp
# import pyautogui

# cam = cv2.VideoCapture(0)
# face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks = True)
# screen_w, screen_h = pyautogui.size()
# while True:
#     _, frame = cam.read()
#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     output = face_mesh.process(rgb_frame)
#     landmark_points = output.multi_face_landmarks
#     frame_h, frame_w, _ = frame.shape
#     if landmark_points:
#         landmarks = landmark_points[0].landmark
#         for id, landmark in enumerate(landmarks[474: 478]):
#             x = int(landmark.x * frame_w)
#             y = int(landmark.y * frame_h)
#             cv2.circle(frame, (x, y), 3, (0, 255, 0))
#             if id == 1:
#                 screen_x = screen_w / frame_w * x
#                 screen_y = screen_h / frame_h * y
#                 pyautogui.moveTo(screen_x, screen_y)
#         left = [landmarks[145], landmarks[159]]
#         for landmark in left:
#             x = int(landmark.x * frame_w)
#             y = int(landmark.y * frame_h)
#             cv2.circle(frame, (x, y), 3, (0, 255, 255))
#         if (left[0].y - left[1].y) < 0.004:
#             pyautogui.click()
#             pyautogui.sleep(1)
# cv2.imshow('Eye Controlled Mouse', frame)
# cv2.waitKey(1)

# import cv2
# import dlib

# # Define the keyboard keys and their coordinates
# keys_set_1 = [['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
#               ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
#               ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';']]

# keys = []
# for i in range(3):
#     row = []
#     for j in range(10):
#         row.append((100 + j * 80, 250 + i * 80))
#     keys.append(row)

# # Define the screen dimensions
# screen_width = 800
# screen_height = 600

# # Define the key dimensions
# key_width = 60
# key_height = 60

# # Define the text string
# text = ""

# # Load the facial landmark predictor
# predictor_path = "shape_predictor_68_face_landmarks.dat"
# predictor = dlib.shape_predictor(predictor_path)

# # Load the face detector
# face_detector = dlib.get_frontal_face_detector()

# # Function to detect eyes in a frame
# def detect_eyes(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_detector(gray)
#     eyes = []
#     for face in faces:
#         landmarks = predictor(gray, face)
#         left_eye = [(landmarks.part(42).x, landmarks.part(42).y),
#                     (landmarks.part(43).x, landmarks.part(43).y),
#                     (landmarks.part(44).x, landmarks.part(44).y),
#                     (landmarks.part(45).x, landmarks.part(45).y),
#                     (landmarks.part(46).x, landmarks.part(46).y),
#                     (landmarks.part(47).x, landmarks.part(47).y)]
#         right_eye = [(landmarks.part(36).x, landmarks.part(36).y),
#                      (landmarks.part(37).x, landmarks.part(37).y),
#                      (landmarks.part(38).x, landmarks.part(38).y),
#                      (landmarks.part(39).x, landmarks.part(39).y),
#                      (landmarks.part(40).x, landmarks.part(40).y),
#                      (landmarks.part(41).x, landmarks.part(41).y)]
#         eyes.append(left_eye)
#         eyes.append(right_eye)
#     return eyes

# # Function to draw the virtual keyboard on the frame
# def draw_keyboard(frame, text):
#     # Draw the keys on the frame
#     for i in range(3):
#         for j in range(10):
#             key_rect = (keys[i][j][0] - key_width // 2, keys[i][j][1] - key_height // 2,
#                         key_width, key_height)
#             if keys_set_1[i][j] in text:
#                 color = (0, 255, 0)
#             else:
#                 color = (0, 0, 255)
#             cv2.rectangle(frame, (keys[i][j][0] - key_width // 2, keys[i][j][1] - key_height // 2),
#                           (keys[i][j][0] + key_width // 2, keys[i][j][1] + key_height // 2), color, 2)
#             cv2.putText(frame, keys_set_1[i][j], (keys[i][j][0] - 10, keys[i][j][1] + 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
#     # Draw the text string on the frame
#     cv2.putText(frame, text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
#     return frame

# # Capture video from the default camera
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Detect eyes in the frame
#     eyes = detect_eyes(frame)

#     # Detect gaze direction based on eye position
#     for eye in eyes:
#         eye_center = ((eye[0][0] + eye[3][0]) // 2, (eye[1][1] + eye[4][1]) // 2)
#         if eye_center[0] < screen_width // 3:
#             text += keys_set_1[0][0]
#         elif eye_center[0] < 2 * screen_width // 3:
#             text += keys_set_1[0][1]
#         else:
#             text += keys_set_1[0][2]

#     # Draw the virtual keyboard on the frame
#     frame = draw_keyboard(frame, text)

#     cv2.imshow("Virtual Keyboard", frame)

#     # Exit the loop on pressing the 'Esc' key
#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import dlib
# import numpy as np
# import math

# # Load face detector and facial landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Define virtual keyboard keys and their corresponding positions
# keyboard = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'Del', 'Done']
# keyboard_positions = [(50, 50), (150, 50), (250, 50), (50, 150), (150, 150), (250, 150),
#                       (50, 250), (150, 250), (250, 250), (150, 350), (250, 350), (350, 350)]

# # Create a dictionary to map gaze direction to virtual keyboard keys
# gaze_to_key_map = {
#     (-1, -1): '1',  # top left
#     (-1, 0): '4',   # middle left
#     (-1, 1): '7',   # bottom left
#     (0, -1): '2',   # top middle
#     (0, 0): '5',    # center
#     (0, 1): '8',    # bottom middle
#     (1, -1): '3',   # top right
#     (1, 0): '6',    # middle right
#     (1, 1): '9'     # bottom right
# }

# # Initialize gaze direction
# gaze_direction = (0, 0)

# # Start the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the grayscale frame
#     faces = detector(gray)

#     for face in faces:
#         # Detect landmarks for the current face
#         landmarks = predictor(gray, face)

#         # Get the eye landmarks
#         left_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
#         right_eye_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

#         # Calculate the gaze direction for each eye
#         left_gaze_direction = ((left_eye_landmarks[3][0] + left_eye_landmarks[0][0]) // 2,
#                                (left_eye_landmarks[3][1] + left_eye_landmarks[0][1]) // 2)
#         right_gaze_direction = ((right_eye_landmarks[3][0] + right_eye_landmarks[0][0]) // 2,
#                                 (right_eye_landmarks[3][1] + right_eye_landmarks[0][1]) // 2)

#         # Calculate the gaze direction vector
#         gaze_vector = (right_gaze_direction[0] - left_gaze_direction[0],
#                        right_gaze_direction[1] - left_gaze_direction[1])

#         # Normalize the gaze direction vector
#         gaze_magnitude = math.sqrt(gaze_vector[0] ** 2 + gaze_vector[1] ** 2)
#         if gaze_magnitude > 0:
#             gaze_direction = (gaze_vector[0] // gaze_magnitude, gaze_vector[1] // gaze_magnitude)
#         else:
#             gaze_direction = (0, 0)
#             # Map gaze direction to virtual keyboard keys
#     key = gaze_to_key_map.get(gaze_direction, '')

#     # Draw virtual keyboard on frame
#     for i, key_pos in enumerate(keyboard_positions):
#         x, y = key_pos
#         cv2.putText(frame, keyboard[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
#         if key == keyboard[i]:
#             cv2.rectangle(frame, (x, y), (x + 80, y + 80), (0, 255, 0), 2)

#     # Display the frame
#     cv2.imshow("Virtual Keyboard", frame)

# # Exit the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import dlib
import pyautogui

# Load face detector model
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load eye detector model
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")

# Load shape predictor model
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define virtual keyboard layout
keyboard = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
    "A", "S", "D", "F", "G", "H", "J", "K", "L", " ",
    "Z", "X", "C", "V", "B", "N", "M", ",", ".", " "
]

# Define gaze direction sensitivity
gaze_sensitivity = 20

# Define gaze direction to virtual keyboard key mapping
gaze_to_key_map = {
    (-1, -1): " ",   # Center
    (-1, 0): "L",    # Left
    (-1, 1): "L",    # Left
    (0, -1): "U",    # Up
    (0, 0): " ",     # Center
    (0, 1): "D",     # Down
    (1, -1): "R",    # Right
    (1, 0): "R",     # Right
    (1, 1): "R"      # Right
}

# Define text to be printed on virtual keyboard
text = ""

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame from video
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract region of interest (ROI) within the face for eye detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect eyes in the ROI
        eyes = eye_detector.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around detected eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            # Detect eye landmarks
            landmarks = shape_predictor(gray, dlib.rectangle(x + ex, y + ey, x + ex + ew, y + ey + eh))
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.int32)

            # Extract left and right eye landmarks
            left_eye_landmarks = landmarks[36:42]
            right_eye_landmarks = landmarks[42:48]

            # Calculate eye centers
            left_eye_center = np.mean(left_eye_landmarks, axis=0).astype(np.int32)
            right_eye_center = np.mean(right_eye_landmarks, axis=0).astype(np.int32)

            # Draw eye centers on frame
            cv2.circle(frame, (x + left_eye_center[0], y + left_eye_center[1]), 2, (0, 0, 255), -1)
            cv2.circle(frame, (x + right_eye_center[0], y + right_eye_center[1]), 2, (0, 0, 255), -1)

            # Calculate gaze direction
            gaze_direction = (np.sign(right_eye_center[0] - left_eye_center[0]), np.sign(right_eye_center[1] - left_eye_center[1]))

            # Map gaze direction to virtual keyboard key
            key = gaze_to_key_map.get(gaze_direction, " ")

            # Display the selected key on frame
            cv2.putText(frame, key, (x + int(w/2) - 10, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Eye Tracking Virtual Keyboard", frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


