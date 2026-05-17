"""
WebSocket endpoints for real-time streaming:
  /ws/training/{job_id}   — streams step metrics from Redis PubSub
  /ws/portfolio/{session_id} — interactive portfolio recalculation
"""
from __future__ import annotations
import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from app.api.deps import get_redis

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/training/{job_id}")
async def training_stream(
    websocket: WebSocket,
    job_id: int,
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
    snapshot_key = f"training:snapshot:{job_id}"

    # Send the latest known state immediately (covers late subscribers who missed PubSub)
    snapshot = await redis_client.get(snapshot_key)
    if snapshot:
        payload = snapshot.decode() if isinstance(snapshot, bytes) else snapshot
        await websocket.send_text(payload)
        try:
            snap_data = json.loads(payload)
            if snap_data.get("type") in ("converged", "error"):
                # Training already finished — nothing left to stream
                return
        except json.JSONDecodeError:
            pass

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


