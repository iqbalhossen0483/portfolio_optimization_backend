"""
WebSocket endpoints for real-time streaming:
  /ws/training/{job_id}   — streams step metrics from Redis PubSub
  /ws/portfolio/{session_id} — interactive portfolio recalculation
"""
from __future__ import annotations
import asyncio
import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from app.api.deps import get_redis

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/training/{job_id}")
async def training_stream(
    websocket: WebSocket,
    job_id: str,
    redis_client=Depends(get_redis),
) -> None:
    """
    Stream live MASAC training metrics to the client.
    Subscribes to Redis PubSub channel `pubsub:training:{job_id}`.

    Messages:
      {"type": "step",      "step": 1500, "entropy": 3.2, ...}
      {"type": "converged", "step": 234100, "final_sharpe": 1.44, ...}
      {"type": "error",     "message": "..."}
    """
    await websocket.accept()
    channel = f"pubsub:training:{job_id}"
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)
    log.info("WS client subscribed to training stream", job_id=job_id)

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            payload = message["data"]
            if isinstance(payload, bytes):
                payload = payload.decode()
            await websocket.send_text(payload)

            # Stop streaming on convergence or error
            try:
                data = json.loads(payload)
                if data.get("type") in ("converged", "error"):
                    break
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        log.info("WS client disconnected from training stream", job_id=job_id)
    except Exception as exc:
        log.error("WS training stream error", job_id=job_id, error=str(exc))
        await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()


@router.websocket("/portfolio/{session_id}")
async def portfolio_stream(
    websocket: WebSocket,
    session_id: str,
    redis_client=Depends(get_redis),
) -> None:
    """
    Interactive portfolio session: client can update hyperparameters and
    receive recomputed panels in real-time.

    Client → Server: {"action": "update_weights", "hyperparams": {"beta": 0.6}}
    Server → Client: {"type": "recomputed", "cooperative": {...}, ...}
    """
    await websocket.accept()
    log.info("WS portfolio session opened", session_id=session_id)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
                continue

            if msg.get("action") == "update_weights":
                # Placeholder: re-run portfolio generation with updated hyperparams
                await websocket.send_text(json.dumps({
                    "type": "recomputed",
                    "session_id": session_id,
                    "message": "Hyperparams updated — recomputation triggered",
                    "hyperparams": msg.get("hyperparams", {}),
                }))

    except WebSocketDisconnect:
        log.info("WS portfolio session closed", session_id=session_id)
    except Exception as exc:
        log.error("WS portfolio session error", session_id=session_id, error=str(exc))
