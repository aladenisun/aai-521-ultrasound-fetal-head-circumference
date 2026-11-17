"""Fetal head circumference demo.

Upload an ultrasound of a fetal head and the app draws the head outline,
measures the head circumference (HC) in mm and estimates how far along the
pregnancy is. It uses the two U-Nets from notebooks 02 and 03 and measures
the same way as notebook 04: fit an ellipse to each model's mask, fix the
overestimate in pixels, then average the two.

Run from the repo folder:
    streamlit run app/streamlit_app.py

The model weights are too big to keep in the repo, so they sit in the
repo's GitHub release. The app downloads them the first time it runs and
keeps them in app/models/ after that.
"""

import os

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import streamlit as st
import torch
from torch import nn

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")      # the two .pt files from Drive go here
SAMPLE_DIR = os.path.join(APP_DIR, "samples")    # optional example scans
SPLITS_CSV = os.path.join(APP_DIR, "..", "data", "splits.csv")
TRAIN_SIZE = 256

# where the weights live online, downloaded once into app/models/
WEIGHTS_URL = ("https://github.com/aladenisun/aai-521-ultrasound-fetal-head-circumference/"
               "releases/download/v1.0/")

# thresholds picked on validation in notebooks 02 and 03
PLAIN_THRESHOLD = 0.4
RESNET_THRESHOLD = 0.5

# a head outline is close to an ellipse. if a mask fills less than this much of
# its own fitted ellipse, it is probably not the skull
MIN_HEAD_SHAPE = 0.85
MIN_AGREE = 0.85   # Dice between the two models' masks

# overestimate in pixels, measured on validation in notebook 04
PLAIN_OFFSET_PX = 7.19
RESNET_OFFSET_PX = 3.27

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---- the two models, same layers as notebooks 02 and 03 so the weights load ----

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, base=32):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottom = ConvBlock(c4, c5)
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(c4 * 2, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        self.out = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottom(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class PretrainedUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # no download needed, the saved weights replace everything
        self.unet = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.unet((x - self.mean) / self.std)


def get_weights(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL + file_name, path, progress=False)
    return path


@st.cache_resource(show_spinner="Loading the models. This only happens once...")
def load_models():
    plain = UNet(in_channels=1, base=32)
    plain.load_state_dict(torch.load(get_weights("unet_plain.pt"), map_location="cpu"))
    resnet = PretrainedUNet()
    resnet.load_state_dict(torch.load(get_weights("unet_resnet34.pt"), map_location="cpu"))
    return plain.eval(), resnet.eval()


@st.cache_data
def load_hc18_info():
    # file name -> (pixel size, doctors HC, split), so HC18 images fill themselves in
    if not os.path.exists(SPLITS_CSV):
        return {}
    df = pd.read_csv(SPLITS_CSV)
    return {r.filename: (r.pixel_size_mm, r.true_hc_mm, r.split) for r in df.itertuples()}


# ---- measuring, same as notebook 04 ----

def predict_mask(model, gray, threshold, channels):
    h, w = gray.shape
    small = cv2.resize(gray, (TRAIN_SIZE, TRAIN_SIZE), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None]
    if channels == 3:
        x = x.repeat(1, 3, 1, 1)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].numpy()
    # back to the original size before measuring, the square resize stretches the head
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob >= threshold).astype(np.uint8)

    # keep only the biggest blob so stray bits elsewhere in the scan don't count
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return mask
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == biggest).astype(np.uint8)


def ellipse_perimeter(a, b):
    # Ramanujan's approximation, a and b are the two semi axes
    h = ((a - b) / (a + b)) ** 2
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))


def hc_pixels(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.nan
    pts = max(contours, key=cv2.contourArea).reshape(-1, 2)

    # skip points on the image border
    h, w = mask.shape
    keep = (pts[:, 0] > 0) & (pts[:, 0] < w - 1) & (pts[:, 1] > 0) & (pts[:, 1] < h - 1)
    pts = pts[keep]
    if len(pts) < 5:
        return np.nan

    _, axes, _ = cv2.fitEllipse(pts)
    return ellipse_perimeter(axes[0] / 2, axes[1] / 2)


def head_shape_score(mask):
    # overlap between the mask and the ellipse fit to it, 1 means a perfect ellipse
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    biggest = max(contours, key=cv2.contourArea)

    # fit on the points off the image border, same as the measurement
    pts = biggest.reshape(-1, 2)
    h, w = mask.shape
    keep = (pts[:, 0] > 0) & (pts[:, 0] < w - 1) & (pts[:, 1] > 0) & (pts[:, 1] < h - 1)
    if keep.sum() < 5:
        return 0.0
    blob = np.zeros_like(mask)
    cv2.drawContours(blob, [biggest], -1, 1, thickness=cv2.FILLED)
    ellipse = np.zeros_like(mask)
    cv2.ellipse(ellipse, cv2.fitEllipse(pts[keep]), 1, thickness=cv2.FILLED)
    both = np.logical_and(blob, ellipse).sum()
    either = np.logical_or(blob, ellipse).sum()
    return both / either if either else 0.0


def hadlock_weeks(hc_mm):
    hc_cm = hc_mm / 10
    return 8.96 + 0.540 * hc_cm + 0.0003 * hc_cm ** 3


def weeks_and_days(weeks):
    w, d = divmod(round(weeks * 7), 7)
    return f"{w} wk {d} d"


def trimester(weeks):
    if weeks < 14:
        return "first"
    if weeks < 28:
        return "second"
    return "third"


def draw_outlines(gray, plain_mask, resnet_mask):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for mask, color in [(plain_mask, (0, 191, 255)), (resnet_mask, (255, 60, 60))]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, color, 2)
    return img


# ---- the page ----

st.set_page_config(page_title="Fetal Head Circumference", layout="wide")
st.title("Fetal Head Circumference from Ultrasound")
st.write("Upload an ultrasound of a fetal head. The app finds the skull, measures the "
         "head circumference and estimates how far along the pregnancy is.")
st.caption("Use the top-down slice through the head that doctors measure on. "
           "Before about 12 weeks the skull hasn't formed yet.")

with st.sidebar:
    st.header("How to use")
    st.write("1. Upload an ultrasound, or pick a sample.")
    st.write("2. Check the pixel size. HC18 images fill it in on their own.")
    st.write("3. Read the results next to the image.")
    st.divider()
    st.caption("Red outline: ResNet-34 U-Net. Blue outline: plain U-Net. "
               "The head circumference is the average of both after the overestimate fix.")
    st.caption("Sample scans are from the HC18 dataset (van den Heuvel et al., 2018).")
    st.caption("AAI-521 Final Project, University of San Diego")

try:
    plain_model, resnet_model = load_models()
except Exception:
    st.error("Could not load the models. Check your internet connection and refresh the page.")
    st.stop()
hc18_info = load_hc18_info()

samples = []
if os.path.isdir(SAMPLE_DIR):
    samples = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))

left, right = st.columns([3, 2])

# pick an image
with left:
    upload = st.file_uploader("Ultrasound image", type=["png", "jpg", "jpeg"])
    pick = "none"
    if samples and upload is None:
        pick = st.selectbox("Or pick a sample", ["none"] + samples)

if upload is not None:
    name = upload.name
    gray = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
elif pick != "none":
    name = pick
    gray = cv2.imread(os.path.join(SAMPLE_DIR, pick), cv2.IMREAD_GRAYSCALE)
else:
    st.info("Upload an image or pick a sample to start." if samples else "Upload an image to start.")
    st.stop()

if gray is None:
    st.error("Could not read that image. Try a PNG or JPG.")
    st.stop()

# HC18 images know their own pixel size and the doctors' HC
known = hc18_info.get(name)
with right:
    pixel_mm = st.number_input(
        "Pixel size (mm per pixel)", min_value=0.01, max_value=1.0, step=0.001, format="%.4f",
        value=float(known[0]) if known else 0.13,
        help="How many mm one pixel covers. It comes from the scanner and changes with the zoom.")
    if known:
        st.caption("Filled in from the HC18 dataset.")
    else:
        st.warning("Not an HC18 file, so check this matches your scan.")

with st.spinner("Finding the head..."):
    plain_mask = predict_mask(plain_model, gray, PLAIN_THRESHOLD, channels=1)
    resnet_mask = predict_mask(resnet_model, gray, RESNET_THRESHOLD, channels=3)

with left:
    st.image(draw_outlines(gray, plain_mask, resnet_mask), caption=name, width="stretch")

# fix the overestimate in pixels, then convert to mm
plain_hc = (hc_pixels(plain_mask) - PLAIN_OFFSET_PX) * pixel_mm
resnet_hc = (hc_pixels(resnet_mask) - RESNET_OFFSET_PX) * pixel_mm

# only trust an outline that is shaped like a head
plain_ok = head_shape_score(plain_mask) >= MIN_HEAD_SHAPE and not np.isnan(plain_hc)
resnet_ok = head_shape_score(resnet_mask) >= MIN_HEAD_SHAPE and not np.isnan(resnet_hc)

# both models have to land on the same head, or the number is not worth showing
both = np.logical_and(plain_mask, resnet_mask).sum()
total = plain_mask.sum() + resnet_mask.sum()
agree = 2 * both / total if total else 0.0

if not (plain_ok and resnet_ok and agree >= MIN_AGREE):
    with right:
        st.error("No measurement for this image. The two models did not find the same "
                 "head shaped outline, so the result would not be reliable.")
        st.write("This usually means the image is not the standard view used to measure the head, "
                 "or it is a very early scan. Before about 12 weeks there is no skull to measure yet.")
    st.stop()

final_hc = (plain_hc + resnet_hc) / 2

with right:
    st.subheader("Results")
    st.metric("Head circumference", f"{final_hc:.1f} mm")

    if final_hc >= 100:
        weeks = hadlock_weeks(final_hc)
        st.metric("Gestational age (Hadlock)", weeks_and_days(weeks))
        st.write(f"That puts it in the **{trimester(weeks)} trimester**.")
    else:
        st.write("Early scan, so no age estimate. Hadlock's formula is built "
                 "for heads of 100 mm and up.")

    if known:
        doctors_hc = known[1]
        st.metric("Doctors' HC", f"{doctors_hc:.1f} mm")
        if known[2] != "test":
            # the models saw this scan while training, so the match will look better than it really is
            st.caption(f"This scan is from our {'training' if known[2] == 'train' else 'validation'} set, "
                       "so the models have already seen it. Use a test set scan for a fair check.")

        # how far off we are, in mm and in days
        diff = final_hc - doctors_hc
        side = "over" if diff > 0 else "under"
        note = f"Ours is {abs(diff):.1f} mm {side} the doctors' measurement"
        if doctors_hc >= 100 and final_hc >= 100:
            days = (hadlock_weeks(final_hc) - hadlock_weeks(doctors_hc)) * 7
            note += f", about {abs(days):.1f} days of gestational age"
        st.write(note + ".")

    with st.expander("Each model on its own"):
        for label, hc, ok in [("ResNet-34 U-Net", resnet_hc, resnet_ok), ("Plain U-Net", plain_hc, plain_ok)]:
            if ok:
                st.write(f"{label}: {hc:.1f} mm")
            else:
                st.write(f"{label}: outline not shaped like a head, left out")
