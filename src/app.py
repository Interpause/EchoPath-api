"""Main app."""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import base64
import logging

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

__all__ = ["create_app"]
log = logging.getLogger(__name__)

MODEL_PATH = "yoloe-26l-seg-pf.pt"
MODEL_CONF = 0.25
MODEL_IOU = 0.45
MODEL_MAX_DET = 100


def create_app():
    """App factory.

    Creating the app within a function prevents mishaps if using multiprocessing.
    """
    app = FastAPI()
    active_websocket: WebSocket | None = None
    websocket_lock = asyncio.Lock()
    model = YOLO(MODEL_PATH)

    def process_image(enc_img: str):
        img_data = base64.b64decode(enc_img)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid jpeg")

        result = model.predict(
            img,
            conf=MODEL_CONF,
            iou=MODEL_IOU,
            max_det=MODEL_MAX_DET,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        coords = boxes.xyxyn.tolist()
        cls_ids = boxes.cls.tolist()
        confs = boxes.conf.tolist()
        names = result.names if isinstance(result.names, dict) else {}

        return [
            {
                "xmin": x1,
                "ymin": y1,
                "xmax": x2,
                "ymax": y2,
                "label": str(names.get(int(cls_id), int(cls_id))),
                "confidence": conf,
            }
            for (x1, y1, x2, y2), cls_id, conf in zip(coords, cls_ids, confs)
        ]

    @app.get("/hello")
    async def hello():
        """Returns a greeting.

        Returns:
            dict: A greeting message.
        """
        log.warning("zzz... 1 more second...")
        await asyncio.sleep(1)
        log.info("...zzz... oh wha...?!")
        return {"message": "Hello, World!"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        nonlocal active_websocket
        await websocket.accept()

        previous_websocket: WebSocket | None = None
        async with websocket_lock:
            previous_websocket = active_websocket
            active_websocket = websocket

        if previous_websocket is not None and previous_websocket is not websocket:
            try:
                await previous_websocket.close(
                    code=1000, reason="Another client connected"
                )
            except RuntimeError:
                # Previous socket may already be disconnected/closing.
                pass

        try:
            while True:
                data = await websocket.receive_json()
                match data:
                    case {"type": "image", "data": enc_img} if isinstance(enc_img, str):
                        try:
                            detections = process_image(enc_img)
                            await websocket.send_json(
                                {"type": "detections", "data": detections}
                            )
                        except ValueError as exc:
                            await websocket.send_json(
                                {"type": "error", "error": str(exc)}
                            )
                    case _:
                        log.warning("Invalid payload: %r", data)
                        await websocket.send_json(
                            {"type": "error", "error": "Invalid payload"}
                        )
        except WebSocketDisconnect:
            pass
        finally:
            async with websocket_lock:
                if active_websocket is websocket:
                    active_websocket = None

    return app
