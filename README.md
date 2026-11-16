# Fetal Head Circumference Measurement from Ultrasound (HC18)

AAI-521 Applied Computer Vision for AI, Final Team Project
University of San Diego

## Overview

Doctors measure fetal head circumference (HC) on ultrasound to track growth and estimate gestational age. Today this is done by hand: a sonographer traces an ellipse around the skull on screen. It takes time and the result varies from person to person.

This project automates that measurement. A U-Net segments the fetal head in each ultrasound image, an ellipse is fit to the predicted mask, and the ellipse circumference is converted from pixels to millimeters using each image's pixel size.

## Pipeline

1. **Preprocessing:** convert the hand-drawn outline annotations into filled head masks, then resize and augment the images.
2. **Segmentation:** train a U-Net from scratch as a baseline, then a U-Net with a pretrained ResNet-34 encoder, and compare the two.
3. **Measurement:** fit an ellipse to each predicted mask, compute its circumference, convert to mm, and correct the small overestimate left by the filled training masks.
4. **Evaluation:** Dice score for segmentation quality, mean absolute error (mm) for head circumference, Bland-Altman agreement with the doctors, and the error converted to gestational age in days.

## Dataset

**HC18 Grand Challenge** (van den Heuvel et al., 2018), available on Zenodo.

- 999 labeled 2D ultrasound images with expert ellipse annotations
- A CSV with each image's pixel size (mm) and the expert head circumference (mm)
- The official test set has no public labels, so we split the 999 labeled images into train (70%), validation (15%), and test (15%) with a fixed random seed. Images of the same fetus always stay in the same split.

The data is not stored in this repo. It lives in the team's shared Google Drive folder, aai-521-hc18.

## Repository Structure

```
notebooks/
  01_data_preparation_and_eda.ipynb   Download, masks, splits, EDA, data loader check
  02_segmentation_model.ipynb         Plain U-Net trained from scratch
  03_pretrained_unet.ipynb            U-Net with a pretrained ResNet-34 encoder, comparison
  04_measurement_analysis.ipynb       HC measurement, overestimate fix, error analysis
src/
  dataset.py                          PyTorch data loader shared by the notebooks
data/
  splits.csv                          Train, validation, and test split used by every notebook
requirements.txt
README.md
```

## How to Run

1. Add a shortcut to the shared aai-521-hc18 folder in your Google Drive (**Shared with me > right-click aai-521-hc18 > Organize > Add shortcut > My Drive**).
2. Open a notebook in Google Colab: **File > Open notebook > GitHub**, then paste this repo's URL.
3. Run the notebooks in order. Each one mounts Google Drive and reads the files the earlier ones saved there.
   - **01:** creates the masks, splits.csv, and EDA figures.
   - **02 and 03:** need a GPU runtime (**Runtime > Change runtime type > T4 GPU**). Each saves its model, predicted masks, and results to Drive.
   - **04:** runs on a regular CPU runtime. It reads the predicted masks from 02 and 03.

## Results

All numbers are on the same 149 held-out test images.

| Model | Dice (test) | HC MAE (mm, test) |
|---|---|---|
| U-Net from scratch | 0.974 | 2.32 |
| U-Net with pretrained encoder | 0.977 | 2.05 |
| Average of both, overestimate corrected | n/a | 1.94 |

For the final measurement:

- Bias of +0.21 mm, with 95% limits of agreement from -5.0 to +5.5 mm
- 95% of test images within 5 mm of the doctors, 62% within 2 mm
- Off by 1.38 days of gestational age on average (heads of 100 mm and up)

## Team

| Member | Role | Responsibilities |
|---|---|---|
| Rajni Massoun | Model Lead | U-Net baseline, pretrained encoder model, tuning, segmentation results |
| Thiago Alvares | Data Lead | Dataset setup, mask creation, splits, EDA, data loader |
| Nisun Alade | Measurement Lead | Ellipse fitting, HC in mm, overestimate correction, evaluation metrics, error analysis |

## References

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet, 327*(8476), 307–310. https://doi.org/10.1016/S0140-6736(86)90837-8

Hadlock, F. P., Deter, R. L., Harrist, R. B., & Park, S. K. (1984). Estimating fetal age: Computer-assisted analysis of multiple fetal growth parameters. *Radiology, 152*(2), 497–501. https://doi.org/10.1148/radiology.152.2.6739822

Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2015)* (pp. 234–241). Springer.

van den Heuvel, T. L. A., de Bruijn, D., de Korte, C. L., & van Ginneken, B. (2018). Automated measurement of fetal head circumference using 2D ultrasound images. *PLOS ONE, 13*(8), e0200412. https://doi.org/10.1371/journal.pone.0200412
