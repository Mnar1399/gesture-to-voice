from flask import Flask, render_template, jsonify
from keras.models import load_model
import numpy as np
import mediapipe as mp
import cv2
import joblib

app = Flask(__name__)

# =========================
# Load model & scaler
# =========================
model = load_model("best_sign_model.keras")
scaler = joblib.load("scaler.save")

# =========================
# Labels
# =========================
labels = {
    "alsalam_kum": ["السلام عليكم", "Peace be upon you"],
    "win_alhamam": ["وين دورة المياه", "Where is the restroom"],
    "win_altawaree": ["وين الطوارئ", "Where is emergency"]
}

class_names = [
    "alsalam_kum",
    "win_alhamam",
    "win_altawaree"
]

# =========================
# MediaPipe
# =========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.3,
            min_tracking_confidence=0.3
)

# =========================
# Camera
# =========================
cap = cv2.VideoCapture(0)

sequence = []

SEQUENCE_LENGTH = 30

last_valid = np.zeros(126)

# =========================
# Normalize
# =========================
def normalize_hand(hand_array):

    hand = hand_array.reshape(21, 3)

    wrist = hand[0]

    hand = hand - wrist

    max_val = np.max(np.abs(hand))

    if max_val > 0:
        hand = hand / max_val

    return hand.flatten()

# =========================
# Home
# =========================

@app.route("/")
def home():
    return render_template("index.html")

# =========================
# Predict
# =========================

@app.route("/predict")
def predict():
    print("Predict route working")
    global sequence
    global last_valid

    ret, frame = cap.read()
    cv2.imwrite("test_frame.jpg", frame)
    if ret:
     cv2.imshow("test", frame)
    cv2.waitKey(1)
         

    if not ret:
        return jsonify({
            "arabic": "...",
            "english": "Camera error"
        })
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    print("Hand detected:", results.multi_hand_landmarks)

    if not results.multi_hand_landmarks:
        return jsonify({
            "arabic": "...",
            "english": "Waiting for gesture"
        })
   
        

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness
    ):

        label = handedness.classification[0].label

        hand_array = []

        for lm in hand_landmarks.landmark:
            hand_array.extend([lm.x, lm.y, lm.z])

        hand_array = np.array(hand_array)

        hand_array = normalize_hand(hand_array)

        if label == "Left":
            left_hand = hand_array

        elif label == "Right":
            right_hand = hand_array

    frame_features = np.concatenate([left_hand, right_hand])

    if np.all(frame_features == 0):
        frame_features = last_valid
    else:
        last_valid = frame_features

    sequence.append(frame_features)

    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    if len(sequence) < SEQUENCE_LENGTH:
        return jsonify({
            "arabic": "...",
            "english": "Reading gesture..."
        })

    input_data = np.array(sequence)

    input_data = scaler.transform(input_data)

    input_data = input_data.reshape(1, 30, 126)

    prediction = model.predict(input_data, verbose=0)[0]

    print("MODEL PREDICTED")
    print(prediction)

    predicted_class = np.argmax(prediction)

    confidence = prediction[predicted_class]

    print("Confidence:", confidence)

    predicted_label = class_names[predicted_class]

    ar = labels[predicted_label][0]

    en = labels[predicted_label][1]

    return jsonify({
        "arabic": ar,
        "english": en
    })
from flask import Response

def generate_frames():

    while True:

        success, frame = cap.read()

        if not success:
            break

        else:

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
if __name__ == "__main__":
    app.run(debug=True)