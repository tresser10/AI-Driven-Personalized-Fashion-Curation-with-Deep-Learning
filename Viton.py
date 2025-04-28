import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Ensure TensorFlow uses GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)

# Initialize Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load Clothing Image (Transparent PNG)
clothing = cv2.imread("dress.png", cv2.IMREAD_UNCHANGED)
if clothing is None:
    raise ValueError("Error loading clothing image. Check file path.")

# Convert to BGRA if missing alpha channel
if clothing.shape[-1] == 3:
    clothing = cv2.cvtColor(clothing, cv2.COLOR_BGR2BGRA)

# Open Webcam
cap = cv2.VideoCapture(0)

# Smoothing parameters
previous_keypoints = None
alpha_smoothing = 0.9  # Increased for smoother motion

def warp_clothing(clothing, body_keypoints, frame):
    """
    Warps the clothing image using Affine Transformation.
    """
    h, w = clothing.shape[:2]

    # Define source points (from clothing image)
    src_pts = np.array([[w//4, 0], [3*w//4, 0], [w//2, h]], dtype=np.float32)  

    # Destination points (on the body)
    dst_pts = np.array([
        body_keypoints['shoulder_left'], 
        body_keypoints['shoulder_right'], 
        body_keypoints['waist_center']
    ], dtype=np.float32)

    # Compute transformation matrix
    matrix = cv2.getAffineTransform(src_pts, dst_pts)

    # Warp clothing using Affine Transform
    warped_clothing = cv2.warpAffine(clothing, matrix, (frame.shape[1], frame.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return warped_clothing

def overlay_clothing(frame, clothing):
    """
    Overlays clothing on the person's frame.
    """
    if clothing.shape[-1] == 3:
        clothing = cv2.cvtColor(clothing, cv2.COLOR_BGR2BGRA)

    # Extract alpha channel for blending
    alpha = clothing[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha
    
    # Convert frame to float for proper blending
    frame_float = frame.astype(np.float32)
    clothing_float = clothing[:, :, :3].astype(np.float32)

    for c in range(3):  # Apply blending for each channel
        frame_float[:clothing.shape[0], :clothing.shape[1], c] = (
            alpha * clothing_float[:, :, c] + alpha_inv * frame_float[:clothing.shape[0], :clothing.shape[1], c]
        )

    return frame_float.astype(np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose detection
    results = pose.process(image)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract keypoints
        current_keypoints = {
            'shoulder_left': (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                              int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])),
            'shoulder_right': (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                               int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])),
            'waist_center': (int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) * frame.shape[1] / 2),
                             int((landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) * frame.shape[0] / 2))
        }

        # Apply smoothing for flicker reduction
        if previous_keypoints is not None:
            for key in current_keypoints:
                current_keypoints[key] = (
                    int(alpha_smoothing * previous_keypoints[key][0] + (1 - alpha_smoothing) * current_keypoints[key][0]),
                    int(alpha_smoothing * previous_keypoints[key][1] + (1 - alpha_smoothing) * current_keypoints[key][1])
                )

        previous_keypoints = current_keypoints  # Update stored keypoints

        # Dynamically resize clothing based on shoulder width
        shoulder_width = abs(current_keypoints['shoulder_right'][0] - current_keypoints['shoulder_left'][0])
        scale_factor = shoulder_width / clothing.shape[1]

        new_width = int(clothing.shape[1] * scale_factor)
        new_height = int(clothing.shape[0] * scale_factor)

        resized_clothing = cv2.resize(clothing, (new_width, new_height))

        # Warp clothing
        warped_clothing = warp_clothing(resized_clothing, current_keypoints, frame)

        # Overlay clothing
        frame = overlay_clothing(frame, warped_clothing)

        # Draw Pose Landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display frame
    cv2.imshow('Virtual Try-On (Improved)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
