import base64
import os
import time
from datetime import datetime, timedelta
import cv2 as cv
import mediapipe as mp
import numpy as np
import pyttsx3
from flask import Flask, render_template, request, jsonify, Response, session, flash, redirect, url_for, get_flashed_messages
from flask_login import UserMixin
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from scipy.spatial import distance as dis
from werkzeug.security import generate_password_hash, check_password_hash
from joblib import load
from PIL import Image
import numpy as np
from datetime import datetime, timezone
import io
import pandas as pd
from recommendation import predict_recommendation_and_tiredness

model = load('./static/random_forest_model_train.joblib')


application = Flask(__name__)
application.config['SECRET_KEY'] = 'aszxqweraaa'
application.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
application.config['UPLOAD_FOLDER'] = 'path_to_your_upload_folder'
socketio = SocketIO(application)
db = SQLAlchemy(application)


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(1000))
    answer = db.Column(db.String(1000))


class StudySession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.Integer)
    eyes_closed = db.Column(db.Integer)
    yawning = db.Column(db.Integer)
    attention_alert = db.Column(db.Integer)
    tiredness = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('study_sessions', lazy=True))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)



class DrivingSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.Integer)
    eyes_closed = db.Column(db.Integer)
    yawning = db.Column(db.Integer)
    attention_alert = db.Column(db.Integer)
    tiredness = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('driving_sessions', lazy=True))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    stars = db.Column(db.Integer, nullable=False)
    text = db.Column(db.String(1000))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('feedbacks', lazy=True))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(150), nullable=False)
    last_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    image = db.Column(db.LargeBinary)
    date_of_birth = db.Column(db.Date)
    tip_utilizator = db.Column(db.String(50), nullable=False)

    @staticmethod
    def set_password(passw):
        return generate_password_hash(passw)

    @staticmethod
    def check_password(password, hashed_password):
        return check_password_hash(hashed_password, password)


class Articles(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    image = db.Column(db.LargeBinary, nullable=False)
    description = db.Column(db.Text, nullable=False)
    link = db.Column(db.String(255), nullable=False)


class Achievements(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('achievements', lazy=True))


class Stats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    sleep_hours = db.Column(db.Integer, nullable=False)
    exercise_done = db.Column(db.Integer)
    diet_quality = db.Column(db.String)
    hydration = db.Column(db.Integer, nullable=False)
    caffeine_intake = db.Column(db.Integer)
    alcohol_intake = db.Column(db.Integer)
    mood = db.Column(db.String)
    medications = db.Column(db.Integer)
    stress_level = db.Column(db.String)
    anxiety_level = db.Column(db.String)
    social_interactions = db.Column(db.Integer)
    relaxation_time = db.Column(db.Integer)
    focus_time = db.Column(db.Integer)
    screen_time = db.Column(db.Integer)
    energy_level = db.Column(db.String)
    notes = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('daily_stats', lazy=True))



with application.app_context():
    db.session.commit()


@application.context_processor
def inject_flashed_messages():
    return dict(get_flashed_messages=get_flashed_messages)


def draw_landmarks(image, outputs, land_mark, color):
    height, width = image.shape[:2]
    for face in land_mark:
        point = outputs.multi_face_landmarks[0].landmark[face]
        point_scale = (int(point.x * width), int(point.y * height))
        cv.circle(image, point_scale, 2, color, 1)


def euclidean_distance(image, top, bottom):
    height, width = image.shape[0:2]
    point1 = int(top.x * width), int(top.y * height)
    point2 = int(bottom.x * width), int(bottom.y * height)
    distance = dis.euclidean(point1, point2)
    return distance


def get_aspect_ratio(image, outputs, top_bottom, left_right):
    landmark = outputs.multi_face_landmarks[0]
    top = landmark.landmark[top_bottom[0]]
    bottom = landmark.landmark[top_bottom[1]]
    top_bottom_dis = euclidean_distance(image, top, bottom)

    left = landmark.landmark[left_right[0]]
    right = landmark.landmark[left_right[1]]
    left_right_dis = euclidean_distance(image, left, right)

    aspect_ratio = left_right_dis / top_bottom_dis
    return aspect_ratio


def extract_eye_landmarks(face_landmarks, eye_landmark_indices):
    eye_landmarks = []
    for indexi in eye_landmark_indices:
        landmark = face_landmarks.landmark[indexi]
        eye_landmarks.append([landmark.x, landmark.y])
    return np.array(eye_landmarks)


def calculate_midpoint(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    midpoint = (sum(x_coords) // len(x_coords), sum(y_coords) // len(y_coords))
    return midpoint


def check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
    left_eye_midpoint = calculate_midpoint(left_eye_points)
    right_eye_midpoint = calculate_midpoint(right_eye_points)
    left_iris_midpoint = calculate_midpoint(left_iris_points)
    right_iris_midpoint = calculate_midpoint(right_iris_points)
    deviation_threshold_horizontal = 2.8
    # deviation_threshold_vertical = 1.99
    return (abs(left_iris_midpoint[0] - left_eye_midpoint[0]) <= deviation_threshold_horizontal
            and abs(right_iris_midpoint[0] - right_eye_midpoint[0]) <= deviation_threshold_horizontal)


def gen_frames():
    STATIC_IMAGE = False
    REFINE_LANDMARKS = True
    MAX_NO_FACES = 1
    DETECTION_CONFIDENCE = 0.6
    TRACKING_CONFIDENCE = 0.6

    COLOR_GREEN = (0, 255, 0)

    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_EYE_TOP_BOTTOM = [386, 374]
    LEFT_EYE_LEFT_RIGHT = [263, 362]

    RIGHT_EYE_TOP_BOTTOM = [159, 145]
    RIGHT_EYE_LEFT_RIGHT = [133, 33]

    UPPER_LOWER_LIPS = [13, 14]
    LEFT_RIGHT_LIPS = [78, 308]

    FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    face_mesh = mp.solutions.face_mesh
    face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                    max_num_faces=MAX_NO_FACES,
                                    refine_landmarks=REFINE_LANDMARKS,
                                    min_detection_confidence=DETECTION_CONFIDENCE,
                                    min_tracking_confidence=TRACKING_CONFIDENCE)

    camera = cv.VideoCapture(0)
    speech = pyttsx3.init()

    frame_count = 0
    min_frame = 15
    min_tolerance = 5.0

    detection_start_time = None
    warning_delay = 2

    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            outputs = face_model.process(image_rgb)

            if outputs.multi_face_landmarks:
                draw_landmarks(frame, outputs, FACE, COLOR_GREEN)

                mesh_points = []
                for p in outputs.multi_face_landmarks[0].landmark:
                    x = int(p.x * img_w)
                    y = int(p.y * img_h)
                    mesh_points.append((x, y))
                mesh_points = np.array(mesh_points)

                left_eye_points = mesh_points[LEFT_EYE]
                right_eye_points = mesh_points[RIGHT_EYE]
                left_iris_points = mesh_points[LEFT_IRIS]
                right_iris_points = mesh_points[RIGHT_IRIS]

                ratio_left_eye = get_aspect_ratio(frame, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
                ratio_right_eye = get_aspect_ratio(frame, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
                ratio = (ratio_left_eye + ratio_right_eye) / 2

                if ratio > min_tolerance:
                    frame_count += 1
                else:
                    frame_count = 0
                if frame_count > min_frame:
                    speech.say('Please wake up')
                    speech.runAndWait()

                ratio_lips = get_aspect_ratio(frame, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
                if ratio_lips < 1.2:
                    speech.say('Please take rest')
                    speech.runAndWait()

                if not check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
                    if detection_start_time is None:
                        detection_start_time = time.time()
                    elif time.time() - detection_start_time >= warning_delay:
                        speech.say('Please pay attention')
                        speech.runAndWait()
                        detection_start_time = None
                else:
                    detection_start_time = None

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


global eyes_closed_count
global yawning_count
global attention_count
global seyes_closed_count
global syawning_count
global sattention_count


def gen_frames_driving():
    STATIC_IMAGE = False
    REFINE_LANDMARKS = True
    MAX_NO_FACES = 1
    DETECTION_CONFIDENCE = 0.6
    TRACKING_CONFIDENCE = 0.6
    eyes_closed_count = 0
    yawning_count = 0
    attention_count = 0
    COLOR_GREEN = (0, 255, 0)

    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_EYE_TOP_BOTTOM = [386, 374]
    LEFT_EYE_LEFT_RIGHT = [263, 362]

    RIGHT_EYE_TOP_BOTTOM = [159, 145]
    RIGHT_EYE_LEFT_RIGHT = [133, 33]

    UPPER_LOWER_LIPS = [13, 14]
    LEFT_RIGHT_LIPS = [78, 308]

    FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    face_mesh = mp.solutions.face_mesh
    face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                    max_num_faces=MAX_NO_FACES,
                                    refine_landmarks=REFINE_LANDMARKS,
                                    min_detection_confidence=DETECTION_CONFIDENCE,
                                    min_tracking_confidence=TRACKING_CONFIDENCE)

    camera = cv.VideoCapture(0)
    speech = pyttsx3.init()

    frame_count = 0
    min_frame = 15
    min_tolerance = 5.0

    detection_start_time = None
    warning_delay = 2


    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            outputs = face_model.process(image_rgb)

            if outputs.multi_face_landmarks:
                draw_landmarks(frame, outputs, FACE, COLOR_GREEN)

                mesh_points = []
                for p in outputs.multi_face_landmarks[0].landmark:
                    x = int(p.x * img_w)
                    y = int(p.y * img_h)
                    mesh_points.append((x, y))
                mesh_points = np.array(mesh_points)

                left_eye_points = mesh_points[LEFT_EYE]
                right_eye_points = mesh_points[RIGHT_EYE]
                left_iris_points = mesh_points[LEFT_IRIS]
                right_iris_points = mesh_points[RIGHT_IRIS]

                ratio_left_eye = get_aspect_ratio(frame, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
                ratio_right_eye = get_aspect_ratio(frame, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
                ratio = (ratio_left_eye + ratio_right_eye) / 2

                if ratio > min_tolerance:
                    frame_count += 1
                else:
                    frame_count = 0
                if frame_count > min_frame:
                    speech.say('Please wake up')
                    speech.runAndWait()
                    eyes_closed_count += 1

                ratio_lips = get_aspect_ratio(frame, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
                if ratio_lips < 1:
                    speech.say('You should take some rest')
                    speech.runAndWait()
                    yawning_count += 1


                if not check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
                    if detection_start_time is None:
                        detection_start_time = time.time()
                    elif time.time() - detection_start_time >= warning_delay:
                        speech.say('Please pay attention')
                        speech.runAndWait()
                        detection_start_time = None
                        attention_count += 1
                else:
                    detection_start_time = None

                socketio.emit('update_alert_counts', {
                    'eyes_closed_count': eyes_closed_count,
                    'yawning_count': yawning_count,
                    'attention_count': attention_count
                })

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def gen_frames_studying():
    STATIC_IMAGE = False
    REFINE_LANDMARKS = True
    MAX_NO_FACES = 1
    DETECTION_CONFIDENCE = 0.6
    TRACKING_CONFIDENCE = 0.6
    seyes_closed_count = 0
    syawning_count = 0
    sattention_count = 0
    COLOR_GREEN = (0, 255, 0)

    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    LEFT_EYE_TOP_BOTTOM = [386, 374]
    LEFT_EYE_LEFT_RIGHT = [263, 362]

    RIGHT_EYE_TOP_BOTTOM = [159, 145]
    RIGHT_EYE_LEFT_RIGHT = [133, 33]

    UPPER_LOWER_LIPS = [13, 14]
    LEFT_RIGHT_LIPS = [78, 308]

    FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
            377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    face_mesh = mp.solutions.face_mesh
    face_model = face_mesh.FaceMesh(static_image_mode=STATIC_IMAGE,
                                    max_num_faces=MAX_NO_FACES,
                                    refine_landmarks=REFINE_LANDMARKS,
                                    min_detection_confidence=DETECTION_CONFIDENCE,
                                    min_tracking_confidence=TRACKING_CONFIDENCE)

    camera = cv.VideoCapture(0)
    speech = pyttsx3.init()

    frame_count = 0
    min_frame = 15
    min_tolerance = 5.0

    detection_start_time = None
    warning_delay = 2

    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            outputs = face_model.process(image_rgb)

            if outputs.multi_face_landmarks:
                draw_landmarks(frame, outputs , FACE, COLOR_GREEN)

                mesh_points = []
                for p in outputs.multi_face_landmarks[0].landmark:
                    x = int(p.x * img_w)
                    y = int(p.y * img_h)
                    mesh_points.append((x, y))
                mesh_points = np.array(mesh_points)

                left_eye_points = mesh_points[LEFT_EYE]
                right_eye_points = mesh_points[RIGHT_EYE]
                left_iris_points = mesh_points[LEFT_IRIS]
                right_iris_points = mesh_points[RIGHT_IRIS]

                ratio_left_eye = get_aspect_ratio(frame, outputs, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
                ratio_right_eye = get_aspect_ratio(frame, outputs, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
                ratio = (ratio_left_eye + ratio_right_eye) / 2

                if ratio > min_tolerance:
                    frame_count += 1
                else:
                    frame_count = 0
                if frame_count > min_frame:
                    speech.say('You should take a break')
                    seyes_closed_count += 1
                    speech.runAndWait()

                ratio_lips = get_aspect_ratio(frame, outputs, UPPER_LOWER_LIPS, LEFT_RIGHT_LIPS)
                if ratio_lips < 1.8:
                    speech.say('You should take some rest')
                    syawning_count += 1
                    speech.runAndWait()

                if not check_iris_in_middle(left_eye_points, left_iris_points, right_eye_points, right_iris_points):
                    if detection_start_time is None:
                        detection_start_time = time.time()
                    elif time.time() - detection_start_time >= warning_delay:
                        speech.say('Attention distracted')
                        speech.runAndWait()
                        sattention_count += 1

                        detection_start_time = None
                else:
                    detection_start_time = None

                socketio.emit('update_alert_counts', {
                    'seyes_closed_count': seyes_closed_count,
                    'syawning_count': syawning_count,
                    'sattention_count': sattention_count
                })

            ret, buffer = cv.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@application.template_filter('to_js_timestamp')
def to_js_timestamp(value):
    return value.strftime('%Y-%m-%d')


@application.route('/end_session', methods=['POST'])
def end_session():
    session_data = request.json
    driving_session = DrivingSession(
        time=session_data['time'],
        eyes_closed=session_data['eyes_closed_count'],
        yawning=session_data['yawning_count'],
        attention_alert=session_data['attention_count'],
        tiredness=session_data['tiredness'],
        user_id=session_data['user_id']
    )
    db.session.add(driving_session)
    db.session.commit()
    return 'Session data saved successfully'


@application.route('/end_ssession', methods=['POST'])
def end_ssession():
    session_data = request.json
    studying_session = StudySession(
        time=session_data['time'],
        eyes_closed=session_data['eyes_closed_count'],
        yawning=session_data['yawning_count'],
        attention_alert=session_data['attention_count'],
        tiredness=session_data['tiredness'],
        user_id=session_data['user_id']
    )
    db.session.add(studying_session)
    db.session.commit()
    return 'Session data saved successfully'


@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return render_template('form.html')
    else:
        feedbacks = Feedback.query.all()
        articles = Articles.query.all()
        decoded_articles = []
        decoded_feedback = []
        for feedback in feedbacks:
            if feedback.user.image:
                decoded_img = f"data:image/jpeg;base64,{base64.b64encode(feedback.user.image).decode('utf-8')}"
            else:
                decoded_img = 'static/images/default_article.png'
            decoded_feedback.append({'feedback': feedback, 'image': decoded_img})
        for article in articles:
            if article.image:
                decoded_image = f"data:image/jpeg;base64,{base64.b64encode(article.image).decode('utf-8')}"
            else:
                decoded_image = 'static/images/default_article.png'
            decoded_articles.append({'article': article, 'image': decoded_image})
        return render_template('form.html', articles=decoded_articles, feedbacks=decoded_feedback)


@application.route('/stream', methods=['GET', 'POST'])
def stream():
    if request.method == 'POST':
        return render_template('streaming.html')
    return render_template('streaming.html')


@application.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@application.route('/login', methods=['GET', 'POST'])
def login():
    return render_template('login.html')


@application.route('/signup', methods=['GET', 'POST'])
def signup():
    return render_template('signup.html')


@application.route('/faquser')
def faquser():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    faq_items = History.query.all()
    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('faq_user.html', user=user, profile_image=profile_image, faq_items=faq_items)
        else:
            default_image = 'static/images/default.png'
            return render_template('faq_user.html', user=user, default_image=default_image, faq_items=faq_items)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


@application.route('/faqadmin')
def faqadmin():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    faq_items = History.query.all()
    if user:
        session['user_id'] = user.id
        if user.image and faq_items.answer:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('faq_admin.html', user=user, profile_image=profile_image, faq_items=faq_items)
        else:
            default_image = 'static/images/default.png'
            return render_template('faq_admin.html', user=user, default_image=default_image, faq_items=faq_items)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


@application.route('/dash_user', methods=['GET', 'POST'])
def dash_user():
    return redirect(url_for('user_dashboard'))


@application.route('/dash_admin', methods=['GET', 'POST'])
def dash_admin():
    return redirect(url_for('admin_dashboard'))


@application.route('/profile')
def profile():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('user_profile.html', user=user, profile_image=profile_image)
        else:
            default_image = 'static/images/default.png'
            return render_template('user_profile.html', user=user, default_image=default_image)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


@application.route('/user_dashboard')
def user_dashboard():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        session['user_id'] = user.id
        if user:
            if user.image:
                profile_image_url = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
                return render_template('dash_user.html', user=user, profile_image_url=profile_image_url)
            else:
                default_image_path = '../claudiatamas/static/images/default.png'
                return render_template('dash_user.html', user=user, default_image_path=default_image_path)
        else:
            flash('User not found', 'error')
            return redirect(url_for('signup'))
    else:
        flash('You must be logged in to access this page', 'error')
        return redirect(url_for('signup'))


@application.route('/admin_dashboard')
def admin_dashboard():
    user_id = session.get('user_id')
    if user_id:
        user = User.query.get(user_id)
        session['user_id'] = user.id
        if user:
            if user.image:
                profile_image_url = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
                return render_template('dash_admin.html', user=user, profile_image_url=profile_image_url)
            else:
                default_image_path = '../static/images/default.png'
                return render_template('dash_admin.html', user=user, default_image_path=default_image_path)
        else:
            flash('User not found', 'error')
            return redirect(url_for('signup'))
    else:
        flash('You must be logged in to access this page', 'error')
        return redirect(url_for('signup'))


@application.route('/signuppage', methods=['POST'])
def signuppage():
    first_name = request.form.get('firstname')
    last_name = request.form.get('lastname')
    email = request.form.get('email')
    password = request.form.get('password')
    cpassword = request.form.get('cpassword')

    if password != cpassword:
        return "Passwords do not match"

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash('This email is already used. Please choose a different one.', 'error')
        return redirect(url_for('signup'))

    password_hashed = User.set_password(password)
    new_user = User(first_name = first_name, last_name=last_name, email=email, password=password_hashed,
                    tip_utilizator='user')
    db.session.add(new_user)
    db.session.commit()

    return render_template('login.html')


@application.route('/signinpage', methods=['POST'])
def signinpage():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and (user.tip_utilizator == 'user') and User.check_password(password, user.password):
            session['user_id'] = user.id
            return redirect(url_for('user_dashboard'))
        elif user and (user.tip_utilizator == 'admin') and User.check_password(password, user.password):
            session['user_id'] = user.id
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid email or password', 'error')

    return render_template('login.html')


@application.route('/logout')
def logout():
    session.clear()

    return render_template('login.html')


@application.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    user_id = request.form.get('user_id')
    user = User.query.get(user_id)

    if not user:
        return "User not found"

    if request.method == 'POST':
        if request.form.get('action') == 'save':
            user.first_name = request.form.get('first_name', user.first_name)
            user.last_name = request.form.get('last_name', user.last_name)

            date_of_birth = request.form.get('date_of_birth')
            user.date_of_birth = datetime.strptime(date_of_birth, '%Y-%m-%d') if date_of_birth else None
            new_email = request.form.get('email', user.email)

            if User.query.filter(User.email == new_email, User.id != user.id).first() is None:
                user.email = new_email
            else:
                flash('Email already used. Please choose a different email.', 'error')
                return redirect(url_for('profile'))

            if 'profile_image' in request.files:
                profile_image = request.files['profile_image']
                if profile_image.filename != '':
                    upload_folder = application.config['UPLOAD_FOLDER']
                    os.makedirs(upload_folder, exist_ok=True)
                    profile_image_path = os.path.join(upload_folder, profile_image.filename)
                    profile_image.save(profile_image_path)
                    user.image = open(profile_image_path, 'rb').read()
                    os.remove(profile_image_path)

            if 'remove_profile_image' in request.form and request.form['remove_profile_image'] == 'on':
                user.image = None

            db.session.commit()
            return redirect(url_for('profile'))

    return redirect(url_for('user_dashboard'))


@application.route('/submit_question', methods=['POST'])
def submit_question():
    data = request.get_json()
    question = data.get('question')
    new_question = History(question=question)
    db.session.add(new_question)
    db.session.commit()

    return jsonify({'success': True})


@application.route('/submit_answer', methods=['POST'])
def submit_answer():
    data = request.json

    indexi = data.get('index')
    answer = data.get('answer')

    question_entry = History.query.filter_by(id=indexi).first()

    if question_entry:
        question_entry.answer = answer
        db.session.commit()

        response = {'status': 'success', 'message': 'Answer submitted successfully'}
    else:
        response = {'status': 'error', 'message': 'Question not found'}

    return jsonify(response)


@application.route('/sessions')
def sessions():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('sessions_user.html', user=user, profile_image=profile_image)
        else:
            default_image = 'static/images/default.png'
            return render_template('sessions_user.html', user=user, default_image=default_image)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye indices as used during training
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


def extract_eye_regions(image, landmarks, eye_indices):
    eye_features = []
    for index in eye_indices:
        landmark = landmarks.landmark[index]
        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
        eye_features.extend([x, y])
    return np.array(eye_features)


def extract_features(image):
    # Convert PIL Image to NumPy array for OpenCV processing
    img_array = np.array(image)
    img_rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)
    features = np.array([])
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye_features = extract_eye_regions(img_rgb, face_landmarks, LEFT_EYE_INDICES)
            right_eye_features = extract_eye_regions(img_rgb, face_landmarks, RIGHT_EYE_INDICES)
            features = np.concatenate((left_eye_features, right_eye_features))
            break  # Assuming only one face per image, so we break after processing the first face.
    return features


@application.route('/driving_session', methods=['POST'])
def driving_session():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    image_data = request.form['imageData'].split(',')[1]
    image_bytes = io.BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes)

    # Convert the image to RGB if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')


    image_features = extract_features(image)

    image_features = np.array(image_features).reshape(1, -1)

    if image_features.shape[1] != 64:
        return redirect(url_for('driving1_session'))

    prediction = model.predict(image_features)

    if prediction[0] == 0:
        flash('Analysis complete: User is currently displaying signs of active engagement.', 'success')
    elif prediction[0] == 1:
        flash('Analysis complete: User is displaying signs of fatigue. Please consider taking a break.', 'warning')

    if user:
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('driving_session.html', user=user, profile_image=profile_image)
        else:
            default_image = 'static/images/default.png'
            return render_template('driving_session.html', user=user, default_image=default_image)
    else:
        flash('User not found', 'error')
        return redirect(url_for('login'))



@application.route('/driving1_session')
def driving1_session():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('predict.html', user=user, profile_image=profile_image)
        else:
            default_image = 'static/images/default.png'
            return render_template('predict.html', user=user, default_image=default_image)
    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))



@application.route('/studying_session')
def studying_session():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('studying_session.html', user=user, profile_image=profile_image)
        else:
            default_image = 'static/images/default.png'
            return render_template('studying_session.html', user=user, default_image=default_image)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


@application.route('/video_feed_driving')
def video_feed_driving():
    return Response(gen_frames_driving(), mimetype='multipart/x-mixed-replace; boundary=frame')


@application.route('/video_feed_studying')
def video_feed_studying():
    return Response(gen_frames_studying(), mimetype='multipart/x-mixed-replace; boundary=frame')


@application.route('/gestionare_users')
def gestionare_users():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    all_users = User.query.all()
    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('gestionare_users.html', user=user, profile_image=profile_image, all_users=all_users)
        else:
            default_image = 'static/images/default.png'
            return render_template('gestionare_users.html', user=user, default_image=default_image, all_users=all_users)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


@application.route('/search_users')
def search_users():
    query = request.args.get('query')
    if query:
        matching_users = User.query.filter((User.first_name.contains(query)) | (User.email.contains(query))).all()
        return render_template('gestionare_users.html', matching_users=matching_users, query=query)
    else:
        return redirect(url_for('gestionare_users'))


@application.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('User deleted successfully', 'success')
    return redirect(url_for('gestionare_users'))


@application.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    user = User.query.get_or_404(user_id)
    if request.method == 'POST':
        if 'first_name' in request.form:
            user.first_name = request.form['first_name']
        if 'last_name' in request.form:
            user.last_name = request.form['last_name']
        if 'email' in request.form:
            user.email = request.form['email']
        if 'date_of_birth' in request.form and request.form['date_of_birth']:
            date_of_birth = request.form.get('date_of_birth')
            user.date_of_birth = datetime.strptime(date_of_birth, '%Y-%m-%d') if date_of_birth else None
        if 'tip_utilizator' in request.form:
            user.tip_utilizator = request.form['tip_utilizator']
        db.session.commit()
        flash('User details updated successfully', 'success')
        return redirect(url_for('gestionare_users'))

    return render_template('edit_user.html', user=user)


@application.route('/cancel_edit', methods=['GET'])
def cancel_edit():
    return redirect(url_for('gestionare_users'))


@application.route('/adduser')
def adduser():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    faq_items = History.query.all()
    if user:
        session['user_id'] = user.id
        if user.image and faq_items.answer:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('add_user.html', user=user, profile_image=profile_image, faq_items=faq_items)
        else:
            default_image = 'static/images/default.png'
            return render_template('add_user.html', user=user, default_image=default_image, faq_items=faq_items)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


@application.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        email = request.form['email']
        date_of_birth = datetime.strptime(request.form['date_of_birth'], '%Y-%m-%d')
        tip_utilizator = request.form['tip_utilizator']
        password = request.form['password']

        new_user = User(first_name=first_name, last_name=last_name, email=email, date_of_birth=date_of_birth, tip_utilizator=tip_utilizator, password=password)
        db.session.add(new_user)
        db.session.commit()
        return render_template('gestionare_users.html')
    return render_template('add_user.html')


def compare_sessions(session1, session2):
    if session1 is None:
        return session2
    if session2 is None:
        return session1
    return session1 if session1.created_at > session2.created_at else session2


@application.route('/user_study_data')
def user_study_data():
    user_id = session.get('user_id')
    if user_id is None:

        return redirect(url_for('login'))

    user = User.query.get(user_id)
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('signup'))

    study_sessions = StudySession.query.filter_by(user_id=user_id) \
        .order_by(StudySession.created_at.desc()) \
        .limit(5).all()

    driving_sessions = DrivingSession.query.filter_by(user_id=user_id) \
        .order_by(DrivingSession.created_at.desc()) \
        .limit(5).all()


    last_study_session = StudySession.query.filter_by(user_id=user_id).order_by(StudySession.created_at.desc()).first()
    last_driving_session = DrivingSession.query.filter_by(user_id=user_id).order_by(DrivingSession.created_at.desc()).first()

    if last_study_session and last_driving_session:
        last_session = compare_sessions(last_study_session, last_driving_session)
    elif last_study_session:
        last_session = last_study_session
    else:
        last_session = last_driving_session
    if user.image:
        profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
        return render_template('user_study_chart.html', user=user, profile_image=profile_image,
                               study_sessions=study_sessions, driving_sessions=driving_sessions, last_session=last_session)
    else:
        default_image = 'static/images/default.png'
        return render_template('user_study_chart.html', user=user, default_image=default_image,
                                study_sessions=study_sessions, driving_sessions=driving_sessions, last_session=last_session)


@application.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    stars = request.form['stars']
    texts = request.form['feedback']
    user_id = request.form['user_id']
    feedback = Feedback(stars=stars, text=texts, user_id=user_id)
    db.session.add(feedback)
    db.session.commit()
    return redirect(url_for('dash_user'))


@application.route('/articles_admin')
def articles_admin():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    articles = Articles.query.all()

    if user:
        session['user_id'] = user.id

        decoded_articles = []
        for article in articles:
            if article.image:
                decoded_image = f"data:image/jpeg;base64,{base64.b64encode(article.image).decode('utf-8')}"
            else:
                decoded_image = 'static/images/default_article.png'
            decoded_articles.append({'article': article, 'image': decoded_image})

        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
        else:
            profile_image = 'static/images/default.png'

        return render_template('articole_admin.html', user=user, profile_image=profile_image, articles=decoded_articles)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))



@application.route('/add_article')
def add_article():
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
            return render_template('add_article.html', user=user, profile_image=profile_image)
        else:
            default_image = 'static/images/default.png'
            return render_template('add_article.html', user=user, default_image=default_image)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


@application.route('/add_articles', methods=['GET', 'POST'])
def add_articles():
    if request.method == 'POST':
        article_title = request.form['article_title']
        description = request.form['description']

        if 'image' in request.files:
            uploaded_image = request.files['image']
            if uploaded_image.filename != '':
                upload_folder = application.config['UPLOAD_FOLDER']
                os.makedirs(upload_folder, exist_ok=True)
                image_path = os.path.join(upload_folder, uploaded_image.filename)
                uploaded_image.save(image_path)
                with open(image_path, 'rb') as f:
                    image_data = f.read()
            else:
                image_data = './static/images/20945485.jpg'
        else:
            image_data = './static/images/20945485.jpg'

        link = request.form['link']

        new_article = Articles(title=article_title, description=description, image=image_data, link=link)

        db.session.add(new_article)
        db.session.commit()
        return redirect(url_for('articles_admin'))
    return render_template('add_articles.html')


@application.route('/cancel_article', methods=['GET'])
def cancel_article():
    return redirect(url_for('articles_admin'))


@application.route('/search_articles')
def search_articles():
    query = request.args.get('query_article')
    if query:
        matching_articles = Articles.query.filter(Articles.title.contains(query)).all()
        return render_template('articole_admin.html', matching_articles=matching_articles, query=query)
    else:
        return redirect(url_for('articles_admin'))


@application.route('/delete_article/<int:article_id>', methods=['POST'])
def delete_article(article_id):
    article = Articles.query.get_or_404(article_id)
    db.session.delete(article)
    db.session.commit()
    flash('Article deleted successfully', 'success')
    return redirect(url_for('articles_admin'))


@application.route('/edit_article/<int:article_id>', methods=['GET', 'POST'])
def edit_article(article_id):
    article = Articles.query.get_or_404(article_id)
    if request.method == 'POST':
        if 'article_title' in request.form:
            article.title = request.form['article_title']
        if 'description' in request.form:
            article.description = request.form['description']
        if 'link' in request.form:
            article.link = request.form['link']

        if 'image' in request.files:
            uploaded_image = request.files['image']
            if uploaded_image.filename != '':
                upload_folder = application.config['UPLOAD_FOLDER']
                os.makedirs(upload_folder, exist_ok=True)
                image_path = os.path.join(upload_folder, uploaded_image.filename)
                uploaded_image.save(image_path)
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                article.image = image_data
                os.remove(image_path)
        db.session.commit()
        flash('Article details updated successfully', 'success')
        return redirect(url_for('articles_admin'))

    return render_template('edit_article.html', article=article)


@application.route('/reviews_admin')
def reviews_admin():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    feedback = Feedback.query.all()

    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
        else:
            profile_image = 'static/images/default.png'

        return render_template('admin_feedback.html', user=user, profile_image=profile_image, feedback=feedback)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


@application.route('/delete_feedback/<int:feedback_id>', methods=['POST'])
def delete_feedback(feedback_id):
    feedback = Feedback.query.get_or_404(feedback_id)
    db.session.delete(feedback)
    db.session.commit()
    flash('Feedback deleted successfully', 'success')
    return redirect(url_for('reviews_admin'))


@application.route('/achievements_admin')
def achievements_admin():
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    unique_achievements = db.session.query(Achievements).group_by(Achievements.title).all()
    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
        else:
            profile_image = 'static/images/default.png'

        return render_template('admin_achievements.html', user=user, profile_image=profile_image, achievements=unique_achievements)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


@application.route('/add_achievement', methods=['GET', 'POST'])
def add_achievement():
    if request.method == 'POST':
        type = request.form['type']
        title = request.form['title']
        users = User.query.all()
        for user in users:
            new_achievement = Achievements(title=title, type=type, completed=False, user=user)
            db.session.add(new_achievement)

        db.session.commit()
        return redirect(url_for('achievements_admin'))

    return render_template('add_achievement.html')


@application.route('/cancel_achievement', methods=['GET'])
def cancel_achievement():
    return redirect(url_for('achievements_admin'))


@application.route('/delete_achievement/<int:achievement_id>', methods=['POST'])
def delete_achievement(achievement_id):
        achievement = Achievements.query.get_or_404(achievement_id)
        achievements_to_delete = Achievements.query.filter_by(title=achievement.title).all()

        for a in achievements_to_delete:
            db.session.delete(a)

        db.session.commit()
        flash('Achievement(s) deleted successfully', 'success')
        return redirect(url_for('achievements_admin'))


@application.route('/edit_achievement/<int:achievement_id>', methods=['GET', 'POST'])
def edit_achievement(achievement_id):
    achievement = Achievements.query.get_or_404(achievement_id)
    if request.method == 'POST':
        if 'title' in request.form:
            achievement.title = request.form['title']
        if 'type' in request.form:
            achievement.type = request.form['type']
        db.session.commit()
        flash('Achievement details updated successfully', 'success')
        return redirect(url_for('achievements_admin'))
    return render_template('edit_achievements.html', achievement=achievement)


@application.route('/achievements_user')
def achievements_user():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    achievements = Achievements.query.filter_by(user_id=user_id).all()
    study_sessions = StudySession.query.filter_by(user_id=user_id).all()
    driving_sessions = DrivingSession.query.filter_by(user_id=user_id).all()

    has_session_without_alerts = False
    for sessions in driving_sessions:
        if sessions.eyes_closed == 0 and sessions.yawning == 0 and sessions.attention_alert == 0:
            has_session_without_alerts = True
            break


    if has_session_without_alerts:
        achievement = Achievements.query.filter_by(user_id=user_id, title="Driving session with no alerts!").first()
        if achievement and not achievement.completed:

            achievement.completed = True
            db.session.commit()

    check_study_achievements1(user_id)
    check_study_achievements2(user_id)
    check_study_achievements3(user_id)
    check_study_achievements4(user_id)
    check_study_achievements5(user_id)
    check_driving_achievements2(user_id)
    check_driving_achievements3(user_id)
    check_driving_achievements4(user_id)
    check_driving_achievements5(user_id)

    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
        else:
            profile_image = 'static/images/default.png'

        return render_template('user_achievements.html', user=user, profile_image=profile_image,
                               achievements=achievements, study_sessions=study_sessions,
                               driving_sessions=driving_sessions, has_session_without_alerts=has_session_without_alerts)

    else:
        flash('User not found', 'error')
        return redirect(url_for('signup'))


def check_study_achievements1(user_id):
    study_sessions = StudySession.query.filter_by(user_id=user_id).all()

    for session in study_sessions:
        session_time = session.time/60
        if session_time >= 25 and session.attention_alert == 0:
            achievement = Achievements.query.filter_by(user_id=user_id, title="25min Studying with No Distraction").first()
            if achievement and not achievement.completed:
                achievement.completed = True
                db.session.commit()


def check_study_achievements2(user_id):
    study_sessions = StudySession.query.filter_by(user_id=user_id).all()

    for session in study_sessions:
        if session.yawning == 0:
            achievement = Achievements.query.filter_by(user_id=user_id, title="No Yawns Study Session").first()
            if achievement and not achievement.completed:
                achievement.completed = True
                db.session.commit()


def check_study_achievements3(user_id):
    study_sessions = StudySession.query.filter_by(user_id=user_id).all()

    for session in study_sessions:
        if session.attention_alert == 0:
            achievement = Achievements.query.filter_by(user_id=user_id, title="Zero Distraction Study Time").first()
            if achievement and not achievement.completed:
                achievement.completed = True
                db.session.commit()


def check_study_achievements4(user_id):
    study_sessions = StudySession.query.filter_by(user_id=user_id).all()

    for session in study_sessions:
        session_time = session.time/60
        if session_time >= 60 and session.yawning == 0:
            achievement = Achievements.query.filter_by(user_id=user_id, title="1h Studying No Yawning").first()
            if achievement and not achievement.completed:
                achievement.completed = True
                db.session.commit()


def check_study_achievements5(user_id):
    study_sessions = StudySession.query.filter_by(user_id=user_id).all()

    for session in study_sessions:
        session_time = session.time/60
        if session_time >= 30 and session.attention_alert == 0:
            achievement = Achievements.query.filter_by(user_id=user_id, title="25min Studying with No Distraction").first()
            if achievement and not achievement.completed:
                achievement.completed = True
                db.session.commit()


def check_driving_achievements2(user_id):
    driving_sessions = DrivingSession.query.filter_by(user_id=user_id).all()

    for session in driving_sessions:
        session_time = session.time/60
        if session_time >= 25 and session.attention_alert == 0:
            achievement = Achievements.query.filter_by(user_id=user_id, title="25min Driving with No Distraction").first()
            if achievement and not achievement.completed:
                achievement.completed = True
                db.session.commit()


def check_driving_achievements3(user_id):
    driving_sessions = DrivingSession.query.filter_by(user_id=user_id).all()

    alert_free_sessions_count = 0
    for session in driving_sessions:
        if session.eyes_closed == 0 and session.yawning == 0 and session.attention_alert == 0:
            alert_free_sessions_count += 1

    if alert_free_sessions_count >= 10:
        achievement = Achievements.query.filter_by(user_id=user_id, title="10 Alert-Free Driving Sessions").first()
        if achievement and not achievement.completed:
            achievement.completed = True
            db.session.commit()


def check_driving_achievements4(user_id):
    driving_sessions = DrivingSession.query.filter_by(user_id=user_id).all()

    for session in driving_sessions:
        session_time = session.time/60
        if session_time >= 60 and session.eyes_closed == 0:
            achievement = Achievements.query.filter_by(user_id=user_id, title="1h Driving No Eyes Closed").first()
            if achievement and not achievement.completed:
                achievement.completed = True
                db.session.commit()


def check_driving_achievements5(user_id):
    driving_sessions = DrivingSession.query.filter_by(user_id=user_id).all()

    for session in driving_sessions:
        session_time = session.time/60
        if session_time >= 30 and session.yawning == 0:
            achievement = Achievements.query.filter_by(user_id=user_id, title="30min Driving No Yawning").first()
            if achievement and not achievement.completed:
                achievement.completed = True
                db.session.commit()


@application.route('/stats_user')
def stats_user():
    user_id = session.get('user_id')
    user = User.query.get(user_id)
    stats = Stats.query.filter_by(user_id=user_id).all()

    if user:
        session['user_id'] = user.id
        if user.image:
            profile_image = f"data:image/jpeg;base64,{base64.b64encode(user.image).decode('utf-8')}"
        else:
            profile_image = 'static/images/default.png'

        one_month_ago = datetime.now() - timedelta(days=14)
        recent_stats = [stat for stat in stats if stat.date >= one_month_ago]

        stats_data = []
        for stat in recent_stats:
            stats_data.append({
                'id': stat.id,

                'start': stat.date.strftime('%Y-%m-%d'),
                'sleep_hours': stat.sleep_hours,
                'exercise_done': stat.exercise_done,
                'diet_quality': stat.diet_quality,
                'hydration': stat.hydration,
                'caffeine_intake': stat.caffeine_intake,
                'alcohol_intake': stat.alcohol_intake,
                'mood': stat.mood,
                'medications': stat.medications,
                'stress_level': stat.stress_level,
                'anxiety_level': stat.anxiety_level,
                'social_interactions': stat.social_interactions,
                'relaxation_time': stat.relaxation_time,
                'focus_time': stat.focus_time,
                'screen_time': stat.screen_time,
                'energy_level': stat.energy_level,
                'notes': stat.notes
            })

            current_date = datetime.now().date()
            stats_for_today = Stats.query.filter(
                Stats.user_id == user_id,
                Stats.date >= current_date,
                Stats.date < current_date + timedelta(days=1)
            ).first()

            if stats_for_today:
                input_data = {
                    'sleep_hours': stats_for_today.sleep_hours,
                    'exercise_done': stats_for_today.exercise_done,
                    'diet_quality': stats_for_today.diet_quality,
                    'hydration': stats_for_today.hydration,
                    'caffeine_intake': stats_for_today.caffeine_intake,
                    'alcohol_intake': stats_for_today.alcohol_intake,
                    'mood': stats_for_today.mood,
                    'medications': stats_for_today.medications,
                    'stress_level': stats_for_today.stress_level,
                    'anxiety_level': stats_for_today.anxiety_level,
                    'social_interaction': stats_for_today.social_interactions,
                    'relaxation_time': stats_for_today.relaxation_time,
                    'focus_time': stats_for_today.focus_time,
                    'screen_time': stats_for_today.screen_time,
                    'energy_level': stats_for_today.energy_level,

                }
                recommendation, tiredness_message = predict_recommendation_and_tiredness(input_data)
                if tiredness_message == "You seem tired. Take some rest and recharge!":
                        flash(f' Tiredness: {tiredness_message} Recommendation: {recommendation}', 'custom_danger')
                else:
                        flash(f'Tiredness: {tiredness_message} Recommendation: {recommendation}', 'custom_info')

    return render_template('user_stats.html', user=user, profile_image=profile_image, stats=stats_data)



@application.route('/fetch-stats-data/<int:event_id>', methods=['GET'])
def fetch_stats_data(event_id):
    user_id = session.get('user_id')
    user = User.query.get(user_id)

    if not user:
        return jsonify({'success': False, 'error': 'User not authenticated'})

    stat = Stats.query.filter_by(user_id=user_id, id=event_id).first()
    if not stat:
        return jsonify({'success': False, 'error': 'Stats data not found for this user and event ID'})

    stats_data = {
        'id': stat.id,
        'start': stat.date.strftime('%Y-%m-%d'),
        'sleep_hours': stat.sleep_hours,
        'exercise_done': stat.exercise_done,
        'diet_quality': stat.diet_quality,
        'hydration': stat.hydration,
        'caffeine_intake': stat.caffeine_intake,
        'alcohol_intake': stat.alcohol_intake,
        'mood': stat.mood,
        'medications': stat.medications,
        'stress_level': stat.stress_level,
        'anxiety_level': stat.anxiety_level,
        'social_interactions': stat.social_interactions,
        'relaxation_time': stat.relaxation_time,
        'focus_time': stat.focus_time,
        'screen_time': stat.screen_time,
        'energy_level': stat.energy_level,
        'notes': stat.notes
    }

    return jsonify({'success': True, 'statsData': stats_data})




@application.route('/submit_daily_stats',  methods=['GET', 'POST'])
def submit_daily_stats():
    user_id = session.get('user_id')

    if request.method == 'POST':

        today_datetime = datetime.now()
        today_date = today_datetime.date()

        today_start = datetime.combine(today_date, datetime.min.time())
        today_end = datetime.combine(today_date, datetime.max.time())

        existing_stats = Stats.query.filter(
            Stats.user_id == user_id,
            Stats.date >= today_start,
            Stats.date <= today_end
        ).first()

        if existing_stats:
            flash('Stats for today already exist. You cannot add more stats for today.', 'warning')
            return redirect(url_for('stats_user'))


        sleep_hours = request.form['sleep_hours']
        exercise_done = request.form['exercise_done'] == 'yes'
        diet_quality = request.form['diet_quality']
        hydration = request.form['hydration']
        caffeine_intake = request.form['caffeine_intake']
        alcohol_intake = request.form['alcohol_intake']
        mood = request.form['mood']
        medications = request.form['medications'] == 'yes'
        stress_level = request.form['stress_level']
        anxiety_level = request.form['anxiety_level']
        social_interactions = request.form['social_interactions'] == 'yes'
        relaxation_time = request.form['relaxation_time']
        focus_time = request.form['focus_time']
        screen_time = request.form['screen_time']
        energy_level = request.form['energy_level']
        notes = request.form['notes']


        new_stats = Stats(
            sleep_hours=sleep_hours,
            exercise_done=exercise_done,
            diet_quality=diet_quality,
            hydration=hydration,
            caffeine_intake=caffeine_intake,
            alcohol_intake=alcohol_intake,
            mood=mood,
            medications=medications,
            stress_level=stress_level,
            anxiety_level=anxiety_level,
            social_interactions=social_interactions,
            relaxation_time=relaxation_time,
            focus_time=focus_time,
            screen_time=screen_time,
            energy_level=energy_level,
            notes=notes,
            user_id=user_id
        )
        db.session.add(new_stats)
        db.session.commit()



        return redirect(url_for('stats_user'))

    return render_template('dash_user.html')


@application.route('/update_daily_stats/<int:eventId>', methods=['GET', 'POST'])
def update_daily_stats(eventId):
    event = Stats.query.get(eventId)

    if event:

        if request.method == 'POST':
            sleep_hours = request.form.get('edit_sleep_hours')
            diet_quality = request.form.get('edit_diet_quality')
            hydration = request.form.get('edit_hydration')
            mood = request.form.get('edit_mood')
            caffeine_intake = request.form.get('edit_caffeine_intake')
            alcohol_intake = request.form.get('edit_alcohol_intake')
            exercise_done = request.form.get('edit_exercise_done') == 'yes'
            medications = request.form.get('edit_medications') == 'yes'
            stress_level = request.form.get('edit_stress_level')
            anxiety_level = request.form.get('edit_anxiety_level')
            energy_level = request.form.get('edit_energy_level')
            relaxation_time = request.form.get('edit_relaxation_time')
            focus_time = request.form.get('edit_focus_time')
            screen_time = request.form.get('edit_screen_time')
            social_interactions = request.form.get('edit_social_interactions') == 'yes'
            notes = request.form.get('edit_notes')


            event.sleep_hours = sleep_hours
            event.diet_quality = diet_quality
            event.hydration = hydration
            event.mood = mood
            event.caffeine_intake = caffeine_intake
            event.alcohol_intake = alcohol_intake
            event.exercise_done = exercise_done
            event.medications = medications
            event.stress_level = stress_level
            event.anxiety_level = anxiety_level
            event.energy_level = energy_level
            event.relaxation_time = relaxation_time
            event.focus_time = focus_time
            event.screen_time = screen_time
            event.social_interactions = social_interactions
            event.notes = notes
            db.session.commit()
            try:
                db.session.commit()
                return redirect(url_for('stats_user'))
            except Exception as e:
                return "An error occurred while updating the event", 500

        return render_template('user_stats.html')


if __name__ == '__main__':
    socketio.run(application, debug=True, allow_unsafe_werkzeug=True)
