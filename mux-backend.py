import os

import httpx
import cv2
import numpy as np
import re
from fastapi import FastAPI, Path
from typing import Annotated
from fastapi.responses import StreamingResponse
import logging
import mediapipe as mp
import face_recognition

# from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%I:%M:%S %p'
)

logger = logging.getLogger(__name__)
# Add this block BEFORE your routes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins, change this to your specific IP/domain in production
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
#     allow_headers=["*"],
# )

DROIDCAM_URL = {
    'camera1': "http://10.64.170.248:4747/video",
    'camera2': "http://10.64.170.163:81/stream"
}

SKIP = 3
frame_count = 0

logger.info("CWD: " + os.getcwd())

model_path = 'blaze_face_short_range.tflite'
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)
detector = FaceDetector.create_from_options(options)

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
    logger.info(f"Warning: Directory {path} not found.\n")


def process_frame(raw_frame: np.ndarray) -> bytes:
    """
    All your Computer Vision logic goes here.
    Input: OpenCV BGR Image
    Output: JPEG-encoded bytes
    """
    rgb_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
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
                if distances[best_match_idx] < 0.6:  # Lower is stricter
                    name = known_face_names[best_match_idx]

            # Draw UI
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(raw_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(raw_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 2. (Optional) Example: Add a simple text overlay
    # cv2.putText(gray, "PROCESSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

    # 3. Re-encode to JPEG
    success, encoded_img = cv2.imencode('.jpg', raw_frame)
    if not success:
        return b""

    return encoded_img.tobytes()


async def frame_generator(cam_id):
    global frame_count
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", DROIDCAM_URL[cam_id]) as response:
            buffer = b""
            async for chunk in response.aiter_bytes():
                buffer += chunk

                # Logic to find the frame boundary
                if b"--dcmjpeg" in buffer:
                    header_end = buffer.find(b"\r\n\r\n")
                    if header_end == -1:
                        continue

                    header_section = buffer[:header_end].decode('utf-8', errors='ignore')
                    content_length_match = re.search(r"Content-Length:\s*(\d+)", header_section)

                    if content_length_match:
                        content_length = int(content_length_match.group(1))
                        start_of_data = header_end + 4
                        end_of_data = start_of_data + content_length

                        if len(buffer) >= end_of_data:
                            # Extract raw bytes
                            jpg_data = buffer[start_of_data:end_of_data]
                            buffer = buffer[end_of_data:]

                            # Decode bytes to OpenCV Mat
                            nparr = np.frombuffer(jpg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                            if frame is not None:
                                # --- CALL EXTERNAL PROCESSING FUNCTION ---
                                processed_bytes = process_frame(frame)

                                if processed_bytes:
                                    yield (b'--dcmjpeg\r\n'
                                           b'Content-Type: image/jpeg\r\n\r\n' +
                                           processed_bytes + b'\r\n')
                            frame_count = (frame_count + 1) % SKIP
                    else:
                        buffer = buffer[header_end + 4:]


@app.get("/stream/{cam_id}")
async def video_feed(cam_id: Annotated[str, Path(title="The ID of the camera to view")]):
    return StreamingResponse(
        frame_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=dcmjpeg"
    )
