import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from argparse import ArgumentParser

from dataset import *
from model import *

DEFAULT_EPOCHS = 10
DEFAULT_OUTPUT = "model"
DEFAULT_BATCH_SIZE = 256


def get_opts():
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir", help="Directory containing train/validation images", type=str, default="."
    )
    parser.add_argument(
        "--train-data",
        help="Training data, CSV of pairs of (path, base64-encoded hash)",
        type=str,
        required=True,
    )
    parser.add_argument("--val-data", help="Validation data", type=str, required=False)
    parser.add_argument("--epochs", help="Training epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--output",
        help="Name of model output (without extension)",
        type=str,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument("--checkpoint-iter", help="Checkpoint frequency", type=int, default=-1)
    parser.add_argument("--batch-size", help="Batch size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--verbose", help="Print intermediate statistics", action="store_true")
    return parser.parse_args()


def main():
    opts = get_opts()
    use_cuda = check_cuda()

    # init model
    model = Model()
    if use_cuda:
        model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # init dataset
    train_data = HashToImage(opts.train_data, opts.data_dir)
    train_dataloader = DataLoader(
        train_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True
    )
    val_data = HashToImage(opts.val_data, opts.data_dir) if opts.val_data else None
    val_dataloader = (
        DataLoader(val_data, batch_size=opts.batch_size, shuffle=True, pin_memory=True)
        if val_data
        else None
    )

    # train
    with tqdm(range(opts.epochs), unit="epoch", total=opts.epochs) as tepochs:
        for epoch in tepochs:
            train_loss = 0
            for data in tqdm(
                train_dataloader, unit="batch", total=len(train_dataloader), leave=False
            ):
                x, y = data
                y = y.type(torch.float32)
                optimizer.zero_grad()
                if use_cuda:
                    x = x.cuda()
                    y = y.cuda()
                y_ = model(x)
                loss = criterion(y_, y)
                loss.backward()
                optimizer.step()
                train_loss += loss
            train_loss = train_loss.item() / len(train_dataloader)
            # save checkpoint
            if opts.checkpoint_iter > 0 and epoch % opts.checkpoint_iter == 0:
                torch.save(model.state_dict(), "{}-epoch{:d}.pt".format(opts.output, epoch))
            # stats
            if opts.verbose:
                tepochs.clear()
                if val_dataloader:
                    val_loss = compute_val_loss(model, criterion, val_dataloader, use_cuda)
                    print(
                        "Epoch {}, train loss: {:.1f}, val loss: {:.1f}".format(
                            epoch, train_loss, val_loss
                        )
                    )
                else:
                    print("Epoch {}, train loss: {:.1f}".format(epoch, train_loss))
            else:
                if val_dataloader:
                    val_loss = compute_val_loss(model, criterion, val_dataloader, use_cuda)
                    tepochs.set_postfix(train_loss=train_loss, val_loss=val_loss)
                else:
                    tepochs.set_postfix(train_loss=train_loss)

    # save final model
    torch.save(model.state_dict(), "{}.pt".format(opts.output))


def compute_val_loss(model, criterion, val_dataloader, use_cuda):
    loss = 0
    model.eval()
    with torch.no_grad():
        for data in val_dataloader:
            x, y = data
            if use_cuda:
                x = x.cuda()
                y = y.cuda()
            y_ = model(x)
            loss += criterion(y_, y).item()
    model.train()
    return loss / len(val_dataloader)


def check_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


if __name__ == "__main__":
    main()
