from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup

from datasets import TweetDataset, create_minibatch
from models import BertClassifier
from utils import timer


def set_seed(seed):
    torch.manual_seed(seed)


def read_data(data_dir: Path):
    train_df = pd.read_csv(data_dir / "train.csv")
    train_data = [(text, label) for text, label in zip(train_df["text"], train_df["target"])]
    train_data, val_data = train_test_split(train_data)
    test_df = pd.read_csv(data_dir / "test.csv")
    return train_data, val_data, (test_df["id"], test_df["text"])


def batch_to_cuda(ipt):
    return {k: v.cuda() for k, v in ipt.items()}


def train(args, model, train_dataloader, val_dataloader):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        [
            {"params": model.bert_model.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters()},
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.train_warmup_steps,
        num_training_steps=len(train_dataloader) * args.train_epochs,
    )

    @timer
    def run_epoch(dataloader, train=False):
        model.train() if train else model.eval()

        loss_history, accu_history = [], []
        with torch.enable_grad() if train else torch.no_grad():
            for x, y in tqdm(
                dataloader, desc="Train |" if train else " Val  |", ncols=120, leave=False
            ):
                if train:
                    optimizer.zero_grad()

                y_hat = model(batch_to_cuda(x)).cpu()
                loss = criterion(y_hat, y)

                if train:
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                loss_history.append(loss.detach())
                accu_history.append(np.mean((torch.round(y_hat) == y).tolist()))
        return np.mean(loss_history), np.mean(accu_history)

    if args.freeze_bert:
        print("=== Freeze Bert Weights ===")
        for param in model.bert_model.parameters():
            param.requires_grad = False

    for epoch in range(1, args.train_epochs + 1):
        print(f"Epoch {epoch:2d} / {args.train_epochs}")

        train_time, (train_loss, train_accu) = run_epoch(train_dataloader, train=True)
        print(f"    Train [{train_time:6.3f}] | loss: {train_loss:.4f} ; accu {train_accu:.4f}")

        val_time, (val_loss, val_accu) = run_epoch(val_dataloader, train=False)
        print(f"    Val   [{val_time:6.3f}] | loss: {val_loss:.4f} ; accu {val_accu:.4f}")

        if epoch == args.unfreeze_bert_after:
            print("=== Unfreeze Bert Weights ===")
            for param in model.bert_model.parameters():
                param.requires_grad = True

        if epoch % args.checkpoint_every == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                args.output_dir / f"checkpoints_{epoch:02d}.pt",
            )


def predict(args, model, test_dataloader):
    predictions = []

    model.eval()
    with torch.no_grad():
        for x in tqdm(test_dataloader, desc="Predicting", ncols=120, leave=False):
            y_hat = model(batch_to_cuda(x)).cpu()
            predictions.append(round(y_hat.item()))

    return predictions


def main(args):
    set_seed(args.seed)

    args.output_dir.mkdir(exist_ok=True, parents=True)

    train_data, val_data, (test_ids, test_data) = read_data(args.data_dir)

    bert_model = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataloader = DataLoader(
        TweetDataset(train_data, bert_tokenizer),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=create_minibatch,
    )
    val_dataloader = DataLoader(
        TweetDataset(val_data, bert_tokenizer),
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        collate_fn=create_minibatch,
    )
    test_dataloader = DataLoader(
        TweetDataset(test_data, bert_tokenizer, has_label=False),
        collate_fn=create_minibatch,
    )

    model = BertClassifier(bert_model, 1).cuda()

    if args.do_train:
        train(args, model, train_dataloader, val_dataloader)
        torch.save(model.state_dict(), args.output_dir / "model_weights.pt")

    if args.do_predict:
        ckpt = torch.load(args.weights_path)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)

        predictions = predict(args, model, test_dataloader)

        with open(args.output_dir / "prediction.csv", "w") as f:
            print("id,target", file=f)
            for i, label in zip(test_ids, predictions):
                print(f"{i},{label}", file=f)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--unfreeze_bert_after", type=int, default=15)

    parser.add_argument("--train_epochs", type=int, default=25)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--train_warmup_steps", type=int, default=200)
    parser.add_argument("--predict_batch_size", type=int, default=1)

    parser.add_argument("--checkpoint_every", type=int, default=5)

    parser.add_argument("--data_dir", type=Path, default="data/")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--weights_path", type=Path)

    parser.add_argument("--seed", type=int, default=0x06902029)
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()

    if not args.weights_path:
        if not args.output_dir:
            args.output_dir = Path(f"outputs/{datetime.now().strftime('%m%d-%H%M%S')}")
        args.weights_path = args.output_dir / "model_weights.pt"
    else:
        args.output_dir = args.weights_path.parent

    return args


if __name__ == "__main__":
    main(parse_arguments())
