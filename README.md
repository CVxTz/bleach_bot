# bleach_bot


## BOT config on discord:
Go to:

https://www.freecodecamp.org/news/create-a-discord-bot-with-python/

Do all the steps needed to get your token and invite your bot to a discord server then come back.

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