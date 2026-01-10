import cv2
import mediapipe as mp
import face_recognition
import os
import numpy as np
import sys
import struct
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#with open('pwd.txt', 'w') as ptx:
#    ptx.write(os.getcwd())

# --- 1. SETUP MEDIAPIPE (IMAGE MODE) ---
# Download: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

sys.stderr.write("CWD:" + os.getcwd())

model_path = 'blaze_face_short_range.tflite'
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)
detector = FaceDetector.create_from_options(options)

# --- 2. PRE-LOAD KNOWN FACES ---
known_face_encodings = []
known_face_names = []
path = "known_faces"

if os.path.exists(path):
    for file in os.listdir(path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = face_recognition.load_image_file(os.path.join(path, file))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(file)[0])
else:
    sys.stderr.write(f"Warning: Directory {path} not found.\n")

# --- 3. BINARY IPC FUNCTIONS ---
def read_image_from_stdin():
    # Read 4-byte Big-Endian length header
    length_bytes = sys.stdin.buffer.read(4)
    if not length_bytes:
        return None
    length = struct.unpack('>I', length_bytes)[0]
    # Read the actual image data
    data = sys.stdin.buffer.read(length)
    return np.frombuffer(data, dtype=np.uint8)

def send_image_to_stdout(image_bytes):
    # Write 4-byte Big-Endian length header
    sys.stdout.buffer.write(struct.pack('>I', len(image_bytes)))
    # Write image data
    sys.stdout.buffer.write(image_bytes)
    sys.stdout.flush()

# --- 4. MAIN LOOP ---
try:
    while True:
        raw_data = read_image_from_stdin()
        if raw_data is None:
            break

        # Decode image (OpenCV uses BGR)
        frame = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # Convert to RGB for MediaPipe and face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Step 1: Detect Faces with MediaPipe (Fast)
        detection_result = detector.detect(mp_image)

        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

                # Boundary safety
                x, y = max(0, x), max(0, y)

                # Step 2: Recognize Faces (Slower)
                # face_recognition format: [(top, right, bottom, left)]
                face_location = [(y, x + w, y + h, x)]
                current_encodings = face_recognition.face_encodings(rgb_frame, face_location)

                name = "Unknown"
                if current_encodings and known_face_encodings:
                    distances = face_recognition.face_distance(known_face_encodings, current_encodings[0])
                    best_match_idx = np.argmin(distances)
                    if distances[best_match_idx] < 0.6: # Lower is stricter
                        name = known_face_names[best_match_idx]

                # Draw UI
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Re-encode and send back
        _, encoded_img = cv2.imencode('.jpg', frame)
        send_image_to_stdout(encoded_img.tobytes())

except Exception as e:
    sys.stderr.write(f"Python Error: {str(e)}\n")
finally:
    detector.close()