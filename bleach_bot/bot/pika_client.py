import json
import logging
import os
import uuid

import pika
import pika.exceptions
from pydantic import BaseModel

# Constants
EXCHANGE_NAME = "direct_exchange"
EXCHANGE_TYPE = "direct"
ROUTING_KEY = "predictor"
RABBIT_URL = os.environ.get(
    "RABBIT_URL", "amqp://localhost?connection_attempts=5&retry_delay=5"
)

# Logger
logger = logging.getLogger("client")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class OutputPredict(BaseModel):
    text: str
    score: float


class PredictorClient(object):
    def __init__(self):
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.setup()
        self.response = None
        self.corr_id = None

    def setup(self):
        self.connection = pika.BlockingConnection(pika.URLParameters(url=RABBIT_URL))

        self.channel = self.connection.channel()
        self.channel.exchange_declare(
            exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE
        )

        result = self.channel.queue_declare(
            queue=f"predictor-{uuid.uuid4().hex}", exclusive=True, auto_delete=True
        )
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True,
        )

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, text) -> OutputPredict:
        self.response = None

        logger.info(f"Sending request: {text}")

        try:
            self.corr_id = str(uuid.uuid4())
            self.channel.basic_publish(
                exchange=EXCHANGE_NAME,
                routing_key=ROUTING_KEY,
                properties=pika.BasicProperties(
                    reply_to=self.callback_queue,
                    correlation_id=self.corr_id,
                ),
                body=text,
            )
            while self.response is None:
                self.connection.process_data_events()
            logger.info(f"Received response : {self.response}")

            dict_response = json.loads(self.response)
            return OutputPredict(**dict_response)

        except pika.exceptions.AMQPError:
            logger.exception("Pika exception")
            self.setup()
            return self.call(text)


if __name__ == "__main__":

    client = PredictorClient()

    print(client.call("This is good"))
