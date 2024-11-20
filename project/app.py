from flask import Flask, Response, render_template
import cv2

app = Flask(__name__)

# Load the Haar cascade classifier
cascade_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Check if the cascade classifier is loaded correctly
if cascade_classifier.empty():
    raise IOError("Error: Cascade Classifier not loaded.")

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        raise IOError("Error: Cannot access the webcam.")

    while True:
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret or frame is None:
            print("Error: Failed to capture frame.")
            break

        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        detections = cascade_classifier.detectMultiScale(gray_frame, 1.3, 5)

        # Draw rectangles around detected faces
        for (x, y, w, h) in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame as a byte stream
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
