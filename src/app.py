"""Main app."""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import base64
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from transformers import pipeline
from ultralytics import YOLOE

__all__ = ["create_app"]
log = logging.getLogger(__name__)

MODEL_PATH = "yoloe-26m-seg.pt"
MODEL_CONF = 0.25
MODEL_IOU = 0.45
MODEL_MAX_DET = 100
TEST_PAGE_PATH = Path(__file__).with_name("test.html")
OBSTACLES = ["person", "table", "chair"]


def create_app():
    """App factory.

    Creating the app within a function prevents mishaps if using multiprocessing.
    """
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    active_websocket: WebSocket | None = None
    websocket_lock = asyncio.Lock()
    model = YOLOE(MODEL_PATH)
    model.set_classes(OBSTACLES)
    depth_pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        dtype=torch.float16,
    )

    def decode_img(enc_img: str):
        img_data = base64.b64decode(enc_img)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid jpeg")
        return img

    def detect_objects(img):
        return model.predict(
            img,
            conf=MODEL_CONF,
            iou=MODEL_IOU,
            max_det=MODEL_MAX_DET,
            # half=True,
            verbose=False,
        )[0]

    def get_normalized_bboxes(result):
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

    def get_segmentation_masks(result, image_shape):
        masks = result.masks
        if masks is None or masks.data is None:
            return []

        mask_data = masks.data
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.detach().cpu().numpy()

        img_h, img_w = image_shape[:2]
        resized_masks = []
        for mask in mask_data:
            bool_mask = mask > 0.5
            if bool_mask.shape != (img_h, img_w):
                bool_mask = (
                    cv2.resize(
                        bool_mask.astype(np.uint8),
                        (img_w, img_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    > 0
                )
            resized_masks.append(bool_mask)

        return resized_masks

    def get_depth(enc_img):
        depth = depth_pipe(enc_img)["depth"]
        depth_map = np.array(depth, dtype=np.float32)

        if depth_map.ndim == 3:
            depth_map = depth_map[..., 0]
        depth_map = 1 - depth_map / 255

        return depth_map

    def get_dist_points(
        detections: list[dict[str, Any]],
        segmentation_masks: list[np.ndarray],
        depth_map: np.ndarray,
    ) -> list[dict[str, Any] | None]:
        dist_points: list[dict[str, Any] | None] = []
        depth_h, depth_w = depth_map.shape[:2]

        for idx, detection in enumerate(detections):
            mask = segmentation_masks[idx] if idx < len(segmentation_masks) else None
            if mask is None:
                dist_points.append(None)
                continue

            ys, xs = np.where(mask)
            if ys.size == 0:
                dist_points.append(None)
                continue

            depths = depth_map[ys, xs]
            valid = np.isfinite(depths)
            if not np.any(valid):
                dist_points.append(None)
                continue

            ys = ys[valid]
            xs = xs[valid]
            depths = depths[valid]

            # "75 percentile nearest" depth proxy: 25th percentile of depth values
            # (nearer pixels are treated as lower depth values).
            target_depth = float(np.percentile(depths, 25))
            point_idx = int(np.argmin(np.abs(depths - target_depth)))
            x_norm = float(
                np.clip(xs[point_idx] / max(1, depth_w - 1), a_min=0.0, a_max=1.0)
            )
            y_norm = float(
                np.clip(ys[point_idx] / max(1, depth_h - 1), a_min=0.0, a_max=1.0)
            )

            dist_points.append(
                {
                    "label": detection["label"],
                    "confidence": detection["confidence"],
                    "distance": float(depths[point_idx]),
                    "x": x_norm,
                    "y": y_norm,
                }
            )

        return dist_points

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

    @app.get("/test", response_class=HTMLResponse)
    async def test_page():
        return TEST_PAGE_PATH.read_text(encoding="utf-8")

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
                            img = await asyncio.to_thread(decode_img, enc_img)
                            results, depth_map = await asyncio.gather(
                                asyncio.to_thread(detect_objects, img),
                                asyncio.to_thread(get_depth, enc_img),
                            )

                            detections = get_normalized_bboxes(results)
                            segmentation_masks = get_segmentation_masks(
                                results, img.shape
                            )
                            dist_points = get_dist_points(
                                detections, segmentation_masks, depth_map
                            )
                            for idx, detection in enumerate(detections):
                                detection["dist_point"] = dist_points[idx]

                            await websocket.send_json(
                                {
                                    "type": "detections",
                                    "data": detections,
                                    "dist_points": dist_points,
                                }
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
