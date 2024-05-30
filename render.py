import csv
import copy
import cv2 as cv
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify, request
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_landmarks, get_args, pre_process_landmark

app = Flask(__name__)

cap_device = 0
cap_width = 640
cap_height = 480
use_static_image_mode = False
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=1,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)
keypoint_classifier = KeyPointClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

recognized_word = ""
running = False
paused = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if running:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({"error": "Recognition is not started"}), 400

@app.route('/start', methods=['POST'])
def start():
    global running, paused
    running = True
    paused = False
    return jsonify({"status": "Recognition started"}), 200

@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    return jsonify({"status": "Recognition stopped"}), 200

@app.route('/pause', methods=['POST'])
def pause():
    global paused
    if running:
        paused = True
        return jsonify({"status": "Recognition paused"}), 200
    else:
        return jsonify({"error": "Recognition is not running"}), 400

@app.route('/resume', methods=['POST'])
def resume():
    global paused
    if running and paused:
        paused = False
        return jsonify({"status": "Recognition resumed"}), 200
    else:
        return jsonify({"error": "Recognition is not paused"}), 400

def gen_frames():
    global recognized_word, running, paused
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    while running:
        success, frame = cap.read()
        if not success or paused:
            continue
        else:
            frame = cv.flip(frame, 1)
            debug_image = copy.deepcopy(frame)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True

            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                    if hand_sign_id != -1:
                        recognized_word = keypoint_classifier_labels[hand_sign_id]

                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(
                        debug_image,
                        handedness,
                        keypoint_classifier_labels[hand_sign_id],
                        recognized_word,
                        debug_image.shape[1],
                        debug_image.shape[0]
                    )

            ret, buffer = cv.imencode('.jpg', debug_image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def draw_info_text(image, handedness, hand_sign_text, recognized_word, image_width, image_height):
    if hand_sign_text:  # Only display text if hand_sign_text is not empty
        info_text = "Predicted Text: " + hand_sign_text
        word_text = "Recognized Word: " + recognized_word

        dark_blue = (139, 0, 0)

        font_scale = min(image_width, image_height) / 1000
        thickness = max(1, int(font_scale * 2))

        info_text_size = cv.getTextSize(info_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        word_text_size = cv.getTextSize(word_text, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

        text_x = max(10, int((image_width - info_text_size[0]) / 2))
        text_y = max(60, int(image_height * 0.1))
        word_x = max(10, int((image_width - word_text_size[0]) / 2))
        word_y = text_y + info_text_size[1] + 10

        # cv.putText(image, info_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, dark_blue, thickness, cv.LINE_AA)
        cv.putText(image, info_text, (word_x, word_y), cv.FONT_HERSHEY_SIMPLEX, font_scale, dark_blue, thickness, cv.LINE_AA)

    return image

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
