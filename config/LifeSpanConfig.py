from contextlib import asynccontextmanager

from fastapi import FastAPI

from config.LoguruConfig import log
from entity.BaseMeta.BaseMeta import database


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("lifespan wakeUp")
    await database.connect()
    yield
    log.info("lifespan shutDown")
    await database.disconnect()