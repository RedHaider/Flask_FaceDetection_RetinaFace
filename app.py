from flask import Flask, render_template, Response, url_for,  redirect
import cv2
from retinaface import RetinaFace

app = Flask(__name__)

# Initialize video capture
video_capture = cv2.VideoCapture(1)  # Use 0 or 1 depending on your camera

def generate_frames():
    video_capture = cv2.VideoCapture(1)

    while True:
        # Read a frame from the camera
        success, frame = video_capture.read()
        if not success:
            break

        # Detect faces in the frame using RetinaFace
        faces = RetinaFace.detect_faces(frame)

        # Draw bounding boxes if faces are detected
        if faces:
            for _, face_data in faces.items():
                x1, y1, x2, y2 = face_data['facial_area']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                landmarks = face_data['landmarks']
                for _, point in landmarks.items():
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the video capture if stream is stopped
    video_capture.release()

#video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

###############################################

#Login
    

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/redirect_to_login')
def redirect_to_login():
    return redirect(url_for('index'))

################################################


#Face Detection
@app.route('/face_detectoin')
def facedetection():
    return render_template('index.html')

@app.route('/redirect_to_facedetection')
def redirect_to_facedetection():
    return redirect(url_for('facedetection'))

###################################################

#Dashboard
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/redirect_to_dashboard')
def redirect_to_dashboard():
    return redirect(url_for('dashboard'))

###################################################

#Attendance
@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

@app.route('/route_to_attendance')
def redirect_to_attendance():
    return redirect(url_for('attendance'))

###################################################

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/to_analytics')
def redirect_to_analytics():
    return redirect(url_for('analytics'))
###################################################


if __name__ == "__main__":
    app.run(port=8080, debug=True)
