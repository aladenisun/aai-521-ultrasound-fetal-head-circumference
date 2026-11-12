# Fetal Head Circumference from Ultrasound (HC18)

Measuring fetal head circumference (HC) from 2D ultrasound images with U-Net
segmentation. AAI-521 final project, University of San Diego.

## Project summary

Head circumference is one of the standard measurements used to track fetal
growth. A sonographer normally draws an ellipse around the skull by hand. This
project automates that step:

1. **Segment** the fetal head in the ultrasound image with a U-Net.
2. **Fit an ellipse** to the predicted head mask.
3. **Convert** the ellipse circumference from pixels to millimeters using the
   pixel size of the scan, and compare it with the sonographer's measurement.

Results are reported as the Dice score of the predicted mask and the mean
absolute difference in millimeters between predicted and true HC.

## Dataset

[HC18 on Zenodo](https://zenodo.org/records/1327317) (CC BY 4.0), from
van den Heuvel et al. (2018).

| File | Contents |
|---|---|
| `training_set.zip` | 999 ultrasound images, each with an `_Annotation.png` outline of the head |
| `training_set_pixel_size_and_HC.csv` | filename, pixel size (mm), head circumference (mm) |
| `test_set.zip`, `test_set_pixel_size.csv` | 335 images without outlines (not used for scoring) |

The official test set has no public outlines, so the train, validation and
test splits used here are all cut from the 999 labeled images.

### Getting the data

Nothing needs to be downloaded by hand. Running
`notebooks/01_data_preparation_and_eda.ipynb` downloads the four files from
Zenodo, checks their MD5 hashes, and unzips them into the data folder.

To do it manually instead, download the four files from the Zenodo page into
`<data folder>/raw/` and unzip the two archives there.

The data is **not** stored in this repository. On Colab the data folder is the
team's shared Google Drive folder (`MyDrive/aai-521-hc18` by default, set by
`DATA_DIR` in the first notebook). Locally it is `data/`, or whatever the
`HC18_DATA_DIR` environment variable points to.

## Repository structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_data_preparation_and_eda.ipynb   download, masks, split, EDA
├── src/
│   └── dataset.py                          PyTorch Dataset and DataLoaders
├── data/                                   local data folder (git-ignored)
│   └── splits.csv                          the shared split (tracked)
└── app/                                    optional Streamlit demo
```

Data folder layout after running notebook 01:

```
<data folder>/
├── raw/training_set/     000_HC.png, 000_HC_Annotation.png, ...
├── masks/                000_HC.png, ...   filled head masks (0 or 255)
├── figures/              EDA figures for the report
└── splits.csv            filename, split, pixel_size_mm, true_hc_mm
```

## How to run

### On Google Colab

1. Open the notebook from GitHub in Colab
   (File > Open notebook > GitHub, paste the repo URL).
2. If the team folder is shared with you, add a shortcut to it in *My Drive*
   so it appears under `/content/drive/MyDrive/`. Edit `DATA_DIR` in the Setup
   cell if the folder name differs.
3. Runtime > Run all. The notebook mounts Drive and clones this repo so that
   `src/dataset.py` can be imported.

### Locally

```bash
git clone https://github.com/aladenisun/aai-521-ultrasound-fetal-head-circumference.git
cd aai-521-ultrasound-fetal-head-circumference
pip install -r requirements.txt
jupyter notebook notebooks/
```

### Notebooks

| Notebook | What it does | Needs | Produces |
|---|---|---|---|
| `01_data_preparation_and_eda.ipynb` | Downloads HC18, converts outlines to filled masks, makes the split, runs the EDA, checks the data loader | internet access | `splits.csv`, `masks/`, `figures/` |

Modeling and measurement notebooks will be added to this table as they land.

## The split

- 70% train, 15% validation, 15% test, random seed 42.
- Split names in `splits.csv` are `train`, `val` and `test`.
- Images whose filenames share the same leading number (for example
  `010_HC.png` and `010_2HC.png`) are kept in the same split, so repeat scans
  of one fetus never sit on both sides of the split. This makes the shares
  approximate rather than exact.
- Everyone should read `splits.csv` rather than re-splitting.

## Data at a glance

- 999 labeled images from 806 fetuses; 164 fetuses have more than one image.
- Split sizes: 700 train, 150 validation, 149 test.
- 975 images are 800 x 540 pixels and 24 are a few pixels off, all 8-bit
  grayscale. The loader resizes everything, so no special handling is needed.
- Pixel size ranges from about 0.05 to 0.39 mm.
- Head circumference ranges from 44 to 346 mm. About 60% of images fall
  between 100 and 200 mm, 18% are under 100 mm and 22% are over 200 mm.
- Only 5 heads are cut off by the image edge. Faint skull edges are mostly
  found in the smallest heads.
- Filled masks give a circumference about 0.7 mm larger than the reference
  value on average, because the mask includes the thickness of the drawn
  outline.

## Using the data loader

```python
import sys
sys.path.insert(0, "src")  # or the path to the cloned repo's src folder
from dataset import get_dataloaders

loaders = get_dataloaders(DATA_DIR, image_size=256, batch_size=16)
images, masks = next(iter(loaders["train"]))
# images: float32 (B, 1, 256, 256) in 0-1
# masks:  float32 (B, 1, 256, 256) with values 0 or 1
```

- Augmentation (horizontal flip, rotation up to 15 degrees, brightness and
  contrast) is applied to the training split only. Flips and rotations are
  applied to the image and mask together. Turn it off with `augment=False`.
- `image_size` takes one number or `(height, width)`, for example 384 or 512.
- `in_channels=3` repeats the grayscale image for RGB pretrained encoders.
- `return_meta=True` adds the filename, pixel size, true HC and original
  image size to each item.

**Note for measurement:** resizing to a square changes the head's shape.
Resize a predicted mask back to the original image size before fitting the
ellipse and multiplying by `pixel_size_mm`.

## Code style

Python code follows [PEP 8](https://peps.python.org/pep-0008/) and is checked
with `flake8`.

## Reference

van den Heuvel, T. L. A., de Bruijn, D., de Korte, C. L., & van Ginneken, B.
(2018). Automated measurement of fetal head circumference using 2D ultrasound
images. *PLoS ONE, 13*(8), e0200412.
https://doi.org/10.1371/journal.pone.0200412
