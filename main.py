import logging

from fastapi import FastAPI, HTTPException, status

from contextlib import asynccontextmanager

from classes import MatchResponse, MatchRequest
from config import settings
from ProcessService import Process

process_service: Process = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global process_service
    process_service = Process(
        settings=settings
    )
    yield
    process_service = None


app = FastAPI(lifespan=lifespan)


@app.post("/match", response_model=MatchResponse)
async def match_endpoint(req: MatchRequest):
    try:
        results = process_service.get_answer(req.messages)
        return MatchResponse(results=results)

    except Exception as e:
        logging.error(f"Ошибка при обработке запроса матчинга: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Внутренняя ошибка сервера при обработке текстового сообщения."
        )