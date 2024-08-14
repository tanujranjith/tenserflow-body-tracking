import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
WIDTH, HEIGHT = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

pose = mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9)
hands = mp_hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.9)

def draw_dots(image, landmarks, color=(255, 0, 0), radius=10):
    for landmark in landmarks:
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), radius, color, -1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pose_results = pose.process(rgb_image)
    hand_results = hands.process(rgb_image)

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        
        head_landmarks = [
            landmarks[mp_pose.PoseLandmark.NOSE],
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_INNER],
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_OUTER],
            landmarks[mp_pose.PoseLandmark.LEFT_EYE_OUTER],
            landmarks[mp_pose.PoseLandmark.RIGHT_EYE_INNER],
            landmarks[mp_pose.PoseLandmark.MOUTH_LEFT],
            landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
        ]
        draw_dots(image, head_landmarks, color=(0, 255, 255))  # Yellow dots for head

        shoulders_landmarks = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        ]
        draw_dots(image, shoulders_landmarks, color=(255, 0, 255))  # Magenta dots for shoulders

        extra_landmarks = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW],
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        ]
        draw_dots(image, extra_landmarks, color=(255, 255, 0))  # Cyan dots for extra points

        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            draw_dots(image, hand_landmarks.landmark, color=(0, 255, 0), radius=10)
            
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Body and Hand Tracking', image)
    cv2.resizeWindow('Body and Hand Tracking', WIDTH, HEIGHT)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
