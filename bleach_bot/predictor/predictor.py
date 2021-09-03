import json
import logging
import os
import time
import math

import numpy as np
import onnxruntime as rt
import pika
from tokenizers import Tokenizer


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Constants
EXCHANGE_NAME = "direct_exchange"
EXCHANGE_TYPE = "direct"
ROUTING_KEY = "predictor"
RABBIT_URL = os.environ.get(
    "RABBIT_URL", "amqp://localhost?connection_attempts=5&retry_delay=5"
)

# Logger
logger = logging.getLogger("predictor")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class Consumer:
    def __init__(
        self,
        inference_session,
        tokenizer,
        routing_key=ROUTING_KEY,
    ):
        self.routing_key = routing_key
        self.consumer_name = f"{routing_key}_consumer"
        self.queue_name = f"{self.routing_key}_queue"
        self.rabbit_host = RABBIT_URL
        self.inference_session = inference_session
        self.tokenizer = tokenizer

    def predict(self, text):
        input_name = self.inference_session.get_inputs()[0].name
        label_name = self.inference_session.get_outputs()[0].name

        ids_l = self.tokenizer.encode(text).ids

        prediction = self.inference_session.run(
            [label_name], {input_name: np.array([ids_l]).astype(np.int32)}
        )[0]

        out_dict = {"score": sigmoid(prediction.item()), "text": text}

        return json.dumps(out_dict)

    def main(self):
        connection = pika.BlockingConnection(pika.URLParameters(url=self.rabbit_host))
        channel = connection.channel()

        channel.queue_declare(queue=self.queue_name, exclusive=False)
        channel.exchange_declare(exchange=EXCHANGE_NAME, exchange_type=EXCHANGE_TYPE)
        channel.queue_bind(
            exchange=EXCHANGE_NAME,
            queue=self.queue_name,
            routing_key=self.routing_key,
        )

        def callback(ch, method, properties, body):
            logger.info(f" [x] {self.consumer_name} Received {body.decode('ascii')}")

            response = self.predict(text=body.decode("ascii"))

            ch.basic_publish(
                exchange="",
                routing_key=properties.reply_to,
                properties=pika.BasicProperties(
                    correlation_id=properties.correlation_id
                ),
                body=response,
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)

        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=self.queue_name, on_message_callback=callback)

        logger.info(
            f" [*] {self.consumer_name} Waiting for messages. To exit press CTRL+C"
        )
        channel.start_consuming()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="toxicity_model.onnx")
    parser.add_argument("--tokenizer_path", default="tokenizer.json")

    args = parser.parse_args()
    model_path = args.model_path

    while True:
        try:
            sess = rt.InferenceSession(model_path)
            tokenizer = Tokenizer.from_file(args.tokenizer_path)
            consumer = Consumer(inference_session=sess, tokenizer=tokenizer)
            consumer.main()
        except Exception as e:
            logger.exception("Exception in worker", e)
        time.sleep(60)
