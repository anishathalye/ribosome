import numpy as np
import torch
from PIL import Image

from argparse import ArgumentParser
import base64

from model import *


DEFAULT_OUTPUT = "output.jpg"
DEFAULT_MODEL = "model.pt"


def get_opts():
    parser = ArgumentParser()
    parser.add_argument("hash", help="Base64-encoded hash", type=str)
    parser.add_argument("--model", help="Model checkpoint", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", help="Output filename", type=str, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    opts = get_opts()

    # load model
    model = Model()
    # for a single image, faster to do this on CPU
    model.load_state_dict(torch.load(opts.model, map_location=torch.device("cpu")))
    model.eval()

    hash_tensor = torch.tensor(np.array(list(base64.b64decode(opts.hash)), dtype=np.uint8))
    with torch.no_grad():
        # batch size 1
        inverted = model(hash_tensor.unsqueeze(0))[0]
        # convert from CHW to HWC and to uint8
        inverted = np.clip(inverted.permute(1, 2, 0).numpy(), 0, 255).astype(np.uint8)
    Image.fromarray(inverted).save(opts.output)


if __name__ == "__main__":
    main()
