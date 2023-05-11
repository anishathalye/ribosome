# Ribosome [![Build Status](https://github.com/anishathalye/ribosome/workflows/CI/badge.svg)](https://github.com/anishathalye/ribosome/actions?query=workflow%3ACI)

Synthesize photos from PhotoDNA.

<p align="center">
<img src="https://github.com/anishathalye/assets/blob/master/ribosome/demo.png" width="400" alt="Ribosome demo">
</p>

See the [blog post] for more information.

## Installation

### Dependencies

You can install Python dependencies using `pip install -r requirements.txt`. If
you want to install the packages manually, here is a list:

- [PyTorch][PyTorch] (`torch`, `torchvision`)
- [NumPy][NumPy] (`numpy`)
- [Pillow][Pillow] (`Pillow`)
- [tqdm][tqdm] (`tqdm`)

### Pre-trained models

Ribosome is released with 4 pre-trained models:

- [`coco-model.pt`](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/coco-model.pt): trained on [COCO] 2017 train images
- [`celeba-model.pt`](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/celeba-model.pt): trained on [CelebA] aligned+cropped images
- [`nsfw-model.pt`](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/nsfw-model.pt): trained on 100K SFW+NSFW images scraped from Reddit
- [`coco+celeba+nsfw-model.pt`](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/coco+celeba+nsfw-model.pt): trained on the combination of the above

Use the models trained on NSFW data at your own risk.

## Usage

### Inference

Use the `infer.py` script to produce images from hashes:

```
python infer.py [--model MODEL] [--output OUTPUT] hash
```

The hash is a base64-encoded string, e.g.
`cVwhQ58OSCEOIwF+AigAkT0GAWdwAQs8o04KGYMfHBUANRUOAycUEFABCh6PABIghDBzCa4RTysQYVcvMDdkMypBPSyNAgRCcTf2AC9PfiYSWDw3KTcxPxM2HSqTDSIsgxJFFA+iihERcU4fHEY4Lj0xhw3QJN4OXQwbIzJjVTsUodIVVy3/FY8I/wcui11O`.

### Training

#### Datasets

Datasets consist of images paired with hashes, in the format of a CSV file with
paths/hashes, and image files in a directory. The CSV file has two colums, path
and hash (no header row). The hash is base64-encoded. Images are 100x100 in
size. After producing such a CSV, it may be convenient to shuffle it and split
it into a training set and validation set.

##### Example dataset

Ribosome includes an example dataset in this format, produced from [COCO]:

- [`coco100x100.tar.gz`](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/coco100x100.tar.gz): image files
- [`coco-train.csv`](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/coco-train.csv): training set hashes
- [`coco-val.csv`](https://github.com/anishathalye/ribosome/releases/download/v1.0.0/coco-val.csv): validation set hashes

##### Preparing a dataset

To produce 100x100 images from an existing dataset, it may be convenient to use
[ImageMagick].

To resize `image.jpg` to 100x100 ignoring the original aspect ratio:

```bash
mogrify -resize '100x100!' image.jpg
```

To resize `image.jpg` to 100x100 by taking a center crop:

```bash
mogrify -resize '100x100^' -gravity Center -extent '100x100' image.jpg
```

You can process files in parallel using `find` / `xargs`, e.g. to convert all
`.jpg` images using 24 threads:

```bash
find . -name '*.jpg' | xargs -n 1 -P 24 mogrify -resize '100x100!'
```

Ribosome does not provide code to compute PhotoDNA hashes, but such code is
available in [pyPhotoDNA].

#### Train a model

Use the `train.py` script to train a model on a dataset:


```
python train.py --train-data TRAIN_DATA ...
```

- `--train-data` is the path to the train data CSV
- Paths in the CSV are interpreted relative to `--data-dir` (or `.` if not supplied)
- `--val-data` is the path to the validation data CSV; if provided, the script
  will report the validation loss after every epoch

See `python train.py --help` for all the options.

## Citation

If you use this implementation in your work, please cite the following:

```bibtex
@misc{athalye2021ribosome,
  author = {Anish Athalye},
  title = {Inverting {PhotoDNA}},
  month = dec,
  year = 2021,
  howpublished = {\url{https://www.anishathalye.com/2021/12/20/inverting-photodna/}},
}
```

## License

Copyright (c) Anish Athalye. Released under the MIT License. See
[LICENSE.md](LICENSE.md) for details.

[blog post]: https://www.anishathalye.com/2021/12/20/inverting-photodna/
[PyTorch]: https://pytorch.org/get-started/locally/
[NumPy]: https://numpy.org/install/
[Pillow]: https://pillow.readthedocs.io/en/stable/installation.html
[tqdm]: https://pypi.org/project/tqdm/
[COCO]: https://cocodataset.org/
[CelebA]: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
[pyPhotoDNA]: https://github.com/jankais3r/pyPhotoDNA
[ImageMagick]: https://imagemagick.org/
