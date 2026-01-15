import asyncio
import logging
import os
import re

import cv2
import face_recognition
import httpx
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

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
    'camera2': "http://192.168.0.183:4747/video",
    'camera1': "http://192.168.0.181:81/stream"
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


@app.websocket("/ws/stream/{cam_id}")
async def websocket_endpoint(websocket: WebSocket, cam_id: str):
    await websocket.accept()

    target_url = DROIDCAM_URL.get(cam_id)
    if not target_url:
        await websocket.close(code=1008)
        return

    SKIP_I = 10
    frame_count = 0
    buffer = b""

    async with httpx.AsyncClient() as client:
        try:
            async with client.stream("GET", target_url, timeout=None) as response:
                async for chunk in response.aiter_bytes():
                    buffer += chunk

                    # 1. Find JPEG markers
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9')

                    # 2. Check if BOTH markers exist and end is after start
                    if start != -1 and end != -1 and end > start:
                        # Extract data
                        jpg_data = buffer[start:end + 2]

                        # Advance buffer: Remove processed frame AND any preceding noise
                        buffer = buffer[end + 2:]

                        # 3. SAFETY CHECK: Ensure jpg_data is not empty before decoding
                        if not jpg_data:
                            continue

                        frame_count += 1

                        if frame_count % SKIP_I == 0:
                            nparr = np.frombuffer(jpg_data, np.uint8)

                            # Additional check on nparr size
                            if nparr.size > 0:
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                                if frame is not None:
                                    processed_bytes = process_frame(frame)
                                    if processed_bytes:
                                        await websocket.send_bytes(processed_bytes)
                                        await asyncio.sleep(0.01)

                    # 4. Emergency Buffer Clear:
                    # If buffer gets too large without finding a frame (e.g., 5MB), clear it.
                    if len(buffer) > 5 * 1024 * 1024:
                        buffer = b""

        except WebSocketDisconnect:
            print(f"Client disconnected from {cam_id}")
        except Exception as e:
            # This catch-all prevents the loop from crashing the whole socket
            print(f"Caught handled error on {cam_id}: {e}")
            # Optional: continue instead of re-raising to keep the socket alive
            pass