"""Main app."""

from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

__all__ = ["create_app"]
log = logging.getLogger(__name__)


def create_app():
    """App factory.

    Creating the app within a function prevents mishaps if using multiprocessing.
    """
    app = FastAPI()
    active_websocket: WebSocket | None = None
    websocket_lock = asyncio.Lock()

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
                data = await websocket.receive_text()
                await websocket.send_text(f"Message text was: {data}")
        except WebSocketDisconnect:
            pass
        finally:
            async with websocket_lock:
                if active_websocket is websocket:
                    active_websocket = None

    return app
