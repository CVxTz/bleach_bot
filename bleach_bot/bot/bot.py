import discord
import os
from threading import local
import logging
import time

from pika_client import PredictorClient

TOKEN = os.getenv("TOKEN")
client = discord.Client()

predictor_client = local()

# Logger
logger = logging.getLogger("bot")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@client.event
async def on_ready():
    logger.info(f"We have logged in as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    logger.info(f"Received message: {message.content[:20]}...")

    if not hasattr(predictor_client, "pika_client"):
        predictor_client.pika_client = PredictorClient()

    pika_client = predictor_client.pika_client

    scored = pika_client.call(message.content)  # Use aio-pika ?
    logger.info(f"Score: {scored.score} for message: {message.content[:20]}...")

    if scored.score > 0.8:
        await message.delete()


client.run(TOKEN)

while True:
    try:
        client.run(TOKEN)
    except Exception as e:
        logger.exception("Exception in bot", e)
    time.sleep(10)
