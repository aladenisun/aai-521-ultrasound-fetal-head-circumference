# Fetal Head Circumference Measurement from Ultrasound (HC18)

AAI-521 Applied Computer Vision for AI, Final Team Project
University of San Diego

## Overview

Doctors measure fetal head circumference (HC) on ultrasound to track growth and estimate gestational age. Today this is done by hand: a sonographer traces an ellipse around the skull on screen. It takes time and the result varies from person to person.

This project automates that measurement. A U-Net segments the fetal head in each ultrasound image, an ellipse is fit to the predicted mask, and the ellipse circumference is converted from pixels to millimeters using each image's pixel size.

## Pipeline

1. **Preprocessing:** convert the hand-drawn outline annotations into filled head masks, then resize and augment the images.
2. **Segmentation:** train a U-Net from scratch as a baseline, then a U-Net with a pretrained encoder, and compare the two.
3. **Measurement:** fit an ellipse to each predicted mask, compute its circumference, and convert to mm.
4. **Evaluation:** Dice score for segmentation quality and mean absolute error (mm) for head circumference.

## Dataset

**HC18 Grand Challenge** (van den Heuvel et al., 2018), available on Zenodo.

- 999 labeled 2D ultrasound images with expert ellipse annotations
- A CSV with each image's pixel size (mm) and the expert head circumference (mm)
- The official test set has no public labels, so we split the 999 labeled images into train (70%), validation (15%), and test (15%) with a fixed random seed

The data is not stored in this repo. Download it from Zenodo into the team's shared Google Drive folder.

## Repository Structure

```
fetal_head_circumference.ipynb   Full project notebook (sections below)
app/
  streamlit_app.py               Optional demo: upload an ultrasound, get the outline and HC
README.md
```

The notebook is organized into sections:

1. Setup: package installs and Google Drive mount
2. Data and EDA: loading, mask creation, splits, exploratory analysis
3. Modeling: U-Net baseline and pretrained encoder U-Net, training and tuning
4. Measurement: ellipse fitting, HC in mm, error analysis
5. Results and Comparison

## How to Run

1. Download HC18 from Zenodo into the shared Google Drive folder.
2. Open the notebook in Google Colab: **File > Open notebook > GitHub**, then paste this repo's URL.
3. Switch to a GPU runtime (**Runtime > Change runtime type > GPU**).
4. Run the Setup section to install packages and mount Google Drive, then run the remaining sections in order.

## Results

| Model | Dice (test) | HC MAE (mm, test) |
|---|---|---|
| U-Net from scratch | TBD | TBD |
| U-Net with pretrained encoder | TBD | TBD |

## Team

| Member | Role | Responsibilities |
|---|---|---|
| Rajni Massoun | Model Lead | U-Net baseline, pretrained encoder model, tuning, segmentation results |
| Thiago Alvares | Data Lead | Dataset setup, mask creation, splits, EDA, data loader |
| Nisun Alade | Measurement Lead | Ellipse fitting, HC in mm, evaluation metrics, error analysis, demo app |

## References

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)* (pp. 234–241). Springer.

van den Heuvel, T. L. A., de Bruijn, D., de Korte, C. L., & van Ginneken, B. (2018). Automated measurement of fetal head circumference using 2D ultrasound images. *PLOS ONE, 13*(8), e0200412. https://doi.org/10.1371/journal.pone.0200412
