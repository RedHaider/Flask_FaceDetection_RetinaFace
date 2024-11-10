from flask import Flask, render_template, Response
import cv2
from retinaface import RetinaFace

app = Flask(__name__)

# Initialize video capture
video_capture = cv2.VideoCapture(1)  # Use 0 or 1 depending on your camera

def generate_frames():
    while True:
        # Read a frame from the camera
        success, frame = video_capture.read()
        if not success:
            break

        # Detect faces in the frame using RetinaFace
        faces = RetinaFace.detect_faces(frame)

        # Check if faces are detected and draw bounding boxes
        if faces:
            for _, face_data in faces.items():
                # Get bounding box coordinates
                x1, y1, x2, y2 = face_data['facial_area']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Get landmarks and draw them
                landmarks = face_data['landmarks']
                for _, point in landmarks.items():
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        # Yield the frame for the video feed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(port=8080, debug=True)
