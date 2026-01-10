import httpx
import cv2
import numpy as np
import re
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add this block BEFORE your routes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins, change this to your specific IP/domain in production
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
#     allow_headers=["*"],
# )

DROIDCAM_URL = "http://192.168.0.183:4747/video"


def process_frame(raw_frame: np.ndarray) -> bytes:
    """
    All your Computer Vision logic goes here.
    Input: OpenCV BGR Image
    Output: JPEG-encoded bytes
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)

    # 2. (Optional) Example: Add a simple text overlay
    # cv2.putText(gray, "PROCESSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)

    # 3. Re-encode to JPEG
    success, encoded_img = cv2.imencode('.jpg', gray)
    if not success:
        return b""

    return encoded_img.tobytes()


async def frame_generator():
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", DROIDCAM_URL) as response:
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
                    else:
                        buffer = buffer[header_end + 4:]


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=dcmjpeg"
    )