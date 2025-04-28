from flask import Flask, render_template, Response, jsonify, request
import numpy as np
import cv2
import os
import random
import cvlib as cv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Global variables to store the current dress and recommendations
current_dress_image = None
recommended_colors = []
current_color_index = 0
last_skin_tone = None
last_gender = None

def suggest_colors(skin_tone, gender):
    """Returns a list of recommended colors based on skin tone and gender."""
    color_recommendations = {
        "light": {
            "man": ["Powder Blue", "Cool Gray", "Navy Blue", "Silver", "Sky Blue"],
            "woman": ["Lavender", "Rose Pink", "Soft Lilac", "Mint Green", "Pearl White"]
        },
        "mid-light": {
            "man": ["Olive Green", "Warm Mustard", "Beige", "Terracotta", "Sandy Brown"],
            "woman": ["Peach", "Blush Pink", "Dusty Rose", "Soft Taupe", "Coral"]
        },
        "mid-dark": {
            "man": ["Emerald Green", "Deep Burgundy", "Mustard Yellow", "Copper Brown"],
            "woman": ["Royal Blue", "Teal", "Rust Red", "Plum", "Warm Caramel"]
        },
        "dark": {
            "man": ["Bright Red", "Electric Blue", "Rich Forest Green", "Deep Mahogany"],
            "woman": ["Gold", "Pure White", "Cobalt Blue", "Fuchsia", "Vibrant Yellow"]
        }
    }
    
    return color_recommendations.get(skin_tone.lower(), {}).get(gender.lower(), [])

def load_dress_image(color, dress_dataset_path="images_compressed"):
    """Loads and returns a dress image matching the given color."""
    color_dir = os.path.join(dress_dataset_path, color.replace(" ", "_").lower())
    if os.path.isdir(color_dir):
        images = [img for img in os.listdir(color_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
        if images:
            img_path = os.path.join(color_dir, random.choice(images))
            dress_img = cv2.imread(img_path)
            if dress_img is not None:
                return dress_img
    return None

def analyze_skin_tone_and_gender(skin_tone_model, gender_model, class_names, dress_dataset_path="images_compressed"):
    """Captures video, detects face, classifies skin tone & gender, and suggests dresses."""
    global current_dress_image, recommended_colors, current_color_index, last_skin_tone, last_gender

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to access webcam.")
        return

    gender_classes = ['man', 'woman']

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video frame.")
            break
        
        faces, _ = cv.detect_face(frame)

        for (x1, y1, x2, y2) in faces:
            face_region = frame[y1:y2, x1:x2]

            if face_region.shape[0] < 10 or face_region.shape[1] < 10:
                continue

            # Skin tone classification
            resized_face = cv2.resize(face_region, (64, 64))
            normalized_face = resized_face / 255.0
            input_data = np.expand_dims(normalized_face, axis=0)
            predictions = skin_tone_model.predict(input_data)
            predicted_skin_tone = class_names[np.argmax(predictions)]

            # Gender classification
            face_crop = cv2.resize(face_region, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            gender_predictions = gender_model.predict(face_crop)[0]
            predicted_gender = gender_classes[np.argmax(gender_predictions)]

            # Reset recommendations if skin tone or gender changes
            if predicted_skin_tone != last_skin_tone or predicted_gender != last_gender:
                recommended_colors = suggest_colors(predicted_skin_tone, predicted_gender)
                current_color_index = 0
                current_dress_image = load_dress_image(recommended_colors[current_color_index]) if recommended_colors else None
                last_skin_tone = predicted_skin_tone
                last_gender = predicted_gender

            # Draw face rectangle and put text labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"Skin Tone: {predicted_skin_tone}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Gender: {predicted_gender}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display dress image if available
            if current_dress_image is not None:
                dress_image = cv2.resize(current_dress_image, (150, 150))  # Resize dress image
                frame[10:160, 10:160] = dress_image  # Overlay dress image on the frame

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    skin_tone_model = load_model("skin_tone_model.h5")
    gender_model = load_model("gender_detection_model.h5")
    class_names = ["dark", "light", "mid-dark", "mid-light"]
    return Response(analyze_skin_tone_and_gender(skin_tone_model, gender_model, class_names),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/next_dress')
def next_dress():
    global current_dress_image, recommended_colors, current_color_index

    if recommended_colors:
        current_color_index = (current_color_index + 1) % len(recommended_colors)
        current_dress_image = load_dress_image(recommended_colors[current_color_index])
        return jsonify({"status": "success", "color": recommended_colors[current_color_index]})
    else:
        return jsonify({"status": "error", "message": "No recommended colors available."})

if __name__ == "__main__":
    app.run(debug=True)