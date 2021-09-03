# bleach_bot
A bot that can detect and automatically delete negative or toxic messages on a discord server.

## BOT config on discord:
Go to:

https://www.freecodecamp.org/news/create-a-discord-bot-with-python/

Do all the steps needed to get your token and invite your bot to a discord server.

Create a .env file with the following content:
```
TOKEN=YOUR_TOKEN
```

## Run:
```
wget https://github.com/CVxTz/bleach_bot/releases/download/v1/toxicity_model.onnx -P ./data/
wget https://github.com/CVxTz/bleach_bot/releases/download/v1/tokenizer.json -P ./data/
docker-compose up --build
```

## Data:
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

## Train the tokenizer
```
python bleach_bot/ml/train_tokenizer.py --files YOUR_TEXT_FILES
```
## Train the model
```
python bleach_bot/ml/train_toxicity_classifier.py --tokenizer data/tokenizer.json \
                                                  --data_path data/train.csv
```
## Convert the model to onnx
```
python bleach_bot/ml/torch_to_onnx.py --tokenizer data/tokenizer.json \
                                      --model_path data/toxicity_model.ckpt \
                                      --output_path data/toxicity_model.onnx
```