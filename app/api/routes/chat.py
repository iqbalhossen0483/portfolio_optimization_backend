from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_chat_service, get_current_user
from app.models.domain import User
from app.models.schemas import ChatRequest, ChatResponse, PortfolioAssetPanel
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Chat with the portfolio advisor",
    description=(
        "Send a natural language message to the MASAC portfolio advisor agent.\n\n"
        "**Example queries:**\n"
        "- *'I have $10,000,000 to allocate. Use Portfolio C.'*\n"
        "- *'What portfolio models have been trained?'*\n"
        "- *'Explain the difference between the cooperative and competitive panels.'*\n\n"
        "On the **first message**, omit `session_id` — the server generates one.\n"
        "Include the returned `session_id` in all follow-up messages to continue the conversation.\n\n"
        "When a portfolio is generated, `panels` contains three topology results "
        "(cooperative, competitive, mixed) each with per-asset weights and allocations.\n"
        "`panels` is `null` for purely conversational messages."
    ),
    responses={
        202: {"description": "Agent response with optional portfolio panels"},
        503: {"description": "Google ADK or Gemini API unavailable"},
    },
    status_code=202,
)
async def chat(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = await service.chat(session_id, request.message, str(user.id))
    except Exception as exc:
        print(f"Error during chat processing: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    panels = None
    pr = result.get("portfolio_result")
    if pr and pr.get("panels"):
        panels = {
            topology: [PortfolioAssetPanel(**asset) for asset in assets]
            for topology, assets in pr["panels"].items()
        }

    return ChatResponse(
        session_id=session_id,
        response=result["response"],
        job_id=pr.get("job_id") if pr else None,
        portfolio_model=pr.get("portfolio_model") if pr else None,
        panels=panels,
    )
