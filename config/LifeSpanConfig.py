from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.LoguruConfig import log
from model.entity.BaseMeta.BaseMeta import database
from scheduler.CleaningSchedule import scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("lifespan wakeUp")
    await database.connect()
    scheduler.start()
    yield
    log.info("lifespan shutDown")
    await database.disconnect()