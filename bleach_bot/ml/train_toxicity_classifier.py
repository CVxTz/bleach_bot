import random
from functools import partial

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from bleach_bot.ml.models import TextBinaryClassifier

MAX_LEN = 512


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, hf_tokenizer):
        self.df = df.reset_index()
        self.positives = self.df.index[self.df["label"] == 1].tolist()
        self.negatives = self.df.index[self.df["label"] == 0].tolist()
        self.n_samples = df.shape[0]
        self.tokenizer = hf_tokenizer

    def __len__(self):
        return self.n_samples // 20  # Sub sample for smaller epochs

    def __getitem__(self, _):
        if random.random() > 0.5:  # Each batch has 50% positives and 50% negatives.
            idx = random.choice(self.positives)
        else:
            idx = random.choice(self.negatives)

        x, y = self.df.loc[idx, "comment_text"], self.df.loc[idx, "label"]

        x = torch.tensor(self.tokenizer.encode(x).ids, dtype=torch.int)

        return x, y


def generate_batch(data_batch, pad_idx):
    X, Y = [], []
    for (x, y) in data_batch:
        X.append(x)
        Y.append(y)

    X = pad_sequence(X, padding_value=pad_idx, batch_first=True)

    X = X[:, :MAX_LEN]

    Y = torch.tensor(Y, dtype=torch.float)

    return X, Y


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--epochs", default=200)
    parser.add_argument("--init_model_path", default=None)
    parser.add_argument("--tokenizer", default="../../data/tokenizer.json")
    parser.add_argument("--data", default="../../data/train.csv")
    parser.add_argument("--out_folder", default="../../data/")

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs

    out_folder = args.out_folder

    init_model_path = args.init_model_path
    tokenizer = Tokenizer.from_file(args.tokenizer)

    data = pd.read_csv(args.data)
    data["comment_text"] = data.comment_text.astype(str)

    data["label"] = (
        data[
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        ].sum(axis=1, skipna=True)
        > 0.5
    ).astype(int)

    print(data[["comment_text", "label"]].head(10))
    print(f"Proportion of positives : {round(data.label.mean() * 100)}%")

    train_val, test = train_test_split(data, test_size=0.1, random_state=1337)
    train, val = train_test_split(train_val, test_size=0.1, random_state=1337)

    train_data = Dataset(df=train, hf_tokenizer=tokenizer)
    valid_data = Dataset(df=val, hf_tokenizer=tokenizer)

    print("len(train_data)", len(train_data))
    print("len(valid_data)", len(valid_data))

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(
            generate_batch,
            pad_idx=tokenizer.token_to_id("[PAD]"),
        ),
    )
    val_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=True,
        collate_fn=partial(
            generate_batch,
            pad_idx=tokenizer.token_to_id("[PAD]"),
        ),
    )

    test_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=10,
        shuffle=False,
        collate_fn=partial(
            generate_batch,
            pad_idx=tokenizer.token_to_id("[PAD]"),
        ),
    )

    model = TextBinaryClassifier(
        vocab_size=tokenizer.get_vocab_size(),
        lr=1e-4,
        dropout=0.1,
    )

    if init_model_path:
        model.load_state_dict(torch.load(init_model_path)["state_dict"])

    logger = TensorBoardLogger(
        save_dir=out_folder,
        name="toxicity_logs",
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        mode="min",
        dirpath=out_folder,
        filename="toxicity_model",
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=test_loader)
