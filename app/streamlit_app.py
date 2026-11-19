"""Fetal head circumference demo.

Upload an ultrasound of a fetal head and the app outlines the skull, measures
the head circumference (HC) in mm, estimates gestational age and places the
result on the INTERGROWTH-21st fetal growth standard. It uses the two U-Nets
from notebooks 02 and 03 and measures the same way as notebook 04: fit an
ellipse to each model's mask, fix the overestimate in pixels, then average.

Run from the repo folder:
    streamlit run app/streamlit_app.py

The model weights are too big to keep in the repo, so they sit in the
repo's GitHub release. The app downloads them the first time it runs and
keeps them in app/models/ after that.
"""

import math
import os

import altair as alt
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import streamlit as st
import torch
from torch import nn

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")
SAMPLE_DIR = os.path.join(APP_DIR, "samples")
SPLITS_CSV = os.path.join(APP_DIR, "..", "data", "splits.csv")
TRAIN_SIZE = 256

# where the weights live online, downloaded once into app/models/
WEIGHTS_URL = ("https://github.com/aladenisun/aai-521-ultrasound-fetal-head-circumference/"
               "releases/download/v1.0/")

# thresholds picked on validation in notebooks 02 and 03
PLAIN_THRESHOLD = 0.4
RESNET_THRESHOLD = 0.5

# quality checks. a head outline is close to an ellipse, and both models
# should land on the same head
MIN_HEAD_SHAPE = 0.85
MIN_AGREE = 0.85

# overestimate in pixels, measured on validation in notebook 04
PLAIN_OFFSET_PX = 7.19
RESNET_OFFSET_PX = 3.27

# test set error in image pixels by head size, from notebook 04 (Table F3).
# the error stays about the same in pixels, so each scan's own pixel size
# turns it into an expected range in mm
PIXEL_ERROR = [(100, 19.3), (200, 13.0), (float("inf"), 15.6)]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

PLAIN_COLOR = (0, 191, 255)
RESNET_COLOR = (255, 60, 60)


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


@st.cache_resource(show_spinner="Loading the segmentation models...")
def load_models():
    plain = UNet(in_channels=1, base=32)
    plain.load_state_dict(torch.load(get_weights("unet_plain.pt"), map_location="cpu"))
    resnet = PretrainedUNet()
    resnet.load_state_dict(torch.load(get_weights("unet_resnet34.pt"), map_location="cpu"))
    return plain.eval(), resnet.eval()


@st.cache_data
def load_hc18_info():
    # file name -> (pixel size, expert HC, split), so HC18 images fill themselves in
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

    # keep only the biggest region so stray bits elsewhere in the scan don't count
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return mask
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == biggest).astype(np.uint8)


def ellipse_perimeter(a, b):
    # Ramanujan's approximation, a and b are the two semiaxes
    h = ((a - b) / (a + b)) ** 2
    return np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))


def outline_points(mask):
    # points of the biggest outline that are not on the image border
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    biggest = max(contours, key=cv2.contourArea)
    pts = biggest.reshape(-1, 2)
    h, w = mask.shape
    keep = (pts[:, 0] > 0) & (pts[:, 0] < w - 1) & (pts[:, 1] > 0) & (pts[:, 1] < h - 1)
    return biggest, pts[keep]


def hc_pixels(mask):
    _, pts = outline_points(mask)
    if pts is None or len(pts) < 5:
        return np.nan
    _, axes, _ = cv2.fitEllipse(pts)
    return ellipse_perimeter(axes[0] / 2, axes[1] / 2)


def head_shape_score(mask):
    # overlap between the mask and the ellipse fit to it, 1 means a perfect ellipse
    biggest, pts = outline_points(mask)
    if pts is None or len(pts) < 5:
        return 0.0
    blob = np.zeros_like(mask)
    cv2.drawContours(blob, [biggest], -1, 1, thickness=cv2.FILLED)
    ellipse = np.zeros_like(mask)
    cv2.ellipse(ellipse, cv2.fitEllipse(pts), 1, thickness=cv2.FILLED)
    both = np.logical_and(blob, ellipse).sum()
    either = np.logical_or(blob, ellipse).sum()
    return both / either if either else 0.0


def dice(mask_a, mask_b):
    total = mask_a.sum() + mask_b.sum()
    return 2 * np.logical_and(mask_a, mask_b).sum() / total if total else 0.0


def expected_error_mm(hc_mm, pixel_mm):
    for limit, px in PIXEL_ERROR:
        if hc_mm < limit:
            return px * pixel_mm


# ---- gestational age and growth ----

def hadlock_weeks(hc_mm):
    # Hadlock et al. (1984), HC in cm
    hc_cm = hc_mm / 10
    return 8.96 + 0.540 * hc_cm + 0.0003 * hc_cm ** 3


def weeks_and_days(weeks):
    w, d = divmod(round(weeks * 7), 7)
    return f"{w}w {d}d"


def trimester(weeks):
    if weeks < 14:
        return "First"
    if weeks < 28:
        return "Second"
    return "Third"


def intergrowth_hc(ga):
    # INTERGROWTH-21st fetal HC standard (Papageorghiou et al., 2014), GA in weeks, HC in mm
    lg = np.log(ga)
    median = -28.2849 + 1.69267 * ga ** 2 - 0.397485 * ga ** 2 * lg
    sd = (1.98735 + 0.0136772 * ga ** 3 - 0.00726264 * ga ** 3 * lg
          + 0.000976253 * ga ** 3 * lg ** 2)
    return median, sd


def centile(hc_mm, ga):
    median, sd = intergrowth_hc(ga)
    z = (hc_mm - median) / sd
    return 100 * 0.5 * (1 + math.erf(z / math.sqrt(2))), z


def ordinal(n):
    n = int(round(n))
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def growth_chart(hc_mm, ga, label):
    weeks = np.arange(14, 40.25, 0.25)
    median, sd = intergrowth_hc(weeks)
    z97 = 1.881
    curve = pd.DataFrame({"Gestational age (weeks)": weeks, "p3": median - z97 * sd,
                          "p50": median, "p97": median + z97 * sd})
    x = alt.X("Gestational age (weeks):Q", scale=alt.Scale(domain=[14, 40]))
    band = alt.Chart(curve).mark_area(opacity=0.22, color=ACCENT).encode(
        x=x, y=alt.Y("p3:Q", title="Head circumference (mm)"), y2="p97:Q")
    mid = alt.Chart(curve).mark_line(color=ACCENT).encode(x=x, y="p50:Q")
    edges = alt.Chart(curve.melt("Gestational age (weeks)", ["p3", "p97"])).mark_line(
        color=ACCENT, strokeDash=[4, 4], opacity=0.7).encode(x=x, y="value:Q", detail="variable:N")
    layers = band + edges + mid
    if ga is not None:
        point = pd.DataFrame({"Gestational age (weeks)": [ga], "HC": [hc_mm], "label": [label]})
        layers += alt.Chart(point).mark_point(size=160, filled=True, color="#FF3C3C").encode(
            x=x, y="HC:Q", tooltip=["label:N", alt.Tooltip("HC:Q", format=".1f")])
    # match the dark page, the chart does not follow the custom colors on its own
    return (layers.properties(height=300, background="transparent")
            .configure_axis(labelColor=TEXT, titleColor=TEXT, gridColor="#2A323D", domainColor="#3A4350")
            .configure_view(stroke=None))


def draw_outlines(gray, plain_mask, resnet_mask):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for mask, color in [(plain_mask, PLAIN_COLOR), (resnet_mask, RESNET_COLOR)]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, color, 2)
    return img


def checks_html(rows):
    # small table for the quality checks, plain HTML so it follows the dark style
    body = ""
    for name, value, limit in rows:
        ok = value >= limit
        body += (f"<tr><td>{name}</td><td class='num'>{value:.3f}</td><td class='num'>≥ {limit:.2f}</td>"
                 f"<td class='{'pass' if ok else 'fail'}'>{'Pass' if ok else 'Fail'}</td></tr>")
    return ("<table class='qc'><tr><th>Check</th><th>Value</th><th>Threshold</th><th>Result</th></tr>"
            + body + "</table>")


# ---- look of the page ----

ACCENT = "#F2C14E"
TEXT = "#E6E9EE"

STYLE = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3 {{ font-family: 'IBM Plex Sans', sans-serif; }}
h1 {{ border-bottom: 2px solid {ACCENT}; padding-bottom: 0.4rem; }}
h3 {{ border-left: 3px solid {ACCENT}; padding-left: 0.6rem; font-size: 1.1rem !important; }}

[data-testid="stMetric"] {{
    background: #12171E; border: 1px solid #222A35; border-left: 3px solid {ACCENT};
    border-radius: 6px; padding: 0.6rem 0.9rem; margin-bottom: 0.4rem;
}}
[data-testid="stMetricValue"] div {{ font-family: 'IBM Plex Mono', monospace; color: {ACCENT}; }}

table.qc {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-bottom: 0.8rem; }}
table.qc th {{ text-align: left; color: #9AA4B2; font-weight: 500; border-bottom: 1px solid #2A323D;
               padding: 0.3rem 0.4rem; }}
table.qc td {{ padding: 0.3rem 0.4rem; border-bottom: 1px solid #1C232D; }}
table.qc td.num {{ font-family: 'IBM Plex Mono', monospace; }}
table.qc td.pass {{ color: #5BD48A; font-weight: 600; }}
table.qc td.fail {{ color: #FF6B6B; font-weight: 600; }}
</style>
"""


# ---- the page ----

# dark theme from inside the app, so there is no separate config file.
# the theme is sent at the start of each run, so the very first run sets it and reruns once
THEME = {"base": "dark", "primaryColor": ACCENT, "backgroundColor": "#0B0F14",
         "secondaryBackgroundColor": "#161C24", "textColor": TEXT}
if st.config.get_option("theme.base") != "dark":
    for key, value in THEME.items():
        st.config.set_option(f"theme.{key}", value)
    st.rerun()

st.set_page_config(page_title="Fetal Head Circumference", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)

st.title("Automated Fetal Head Circumference")
st.write("Upload a fetal head ultrasound. The app segments the skull, measures the head "
         "circumference, estimates gestational age and places the result on a fetal growth standard.")
st.caption("Use the standard transthalamic plane used for HC measurement. Scans before about 12 weeks "
           "are not supported, since the skull has not yet ossified.")
st.caption("Research demonstration only. Not for clinical use.")

with st.sidebar:
    st.header("How to use")
    st.write("1. Upload an ultrasound, or select a sample.")
    st.write("2. Confirm the pixel size. HC18 images load it automatically.")
    st.write("3. Optionally, enter the gestational age from a dating scan to get an HC percentile.")
    st.write("4. Review the quality checks and results.")
    st.divider()
    st.caption("Red outline: ResNet-34 U-Net. Blue outline: plain U-Net. The reported HC is the "
               "average of both after bias correction.")

    with st.expander("Method"):
        st.markdown(
            "1. **Segmentation.** Two U-Nets (Ronneberger et al., 2015), one trained from scratch and "
            "one with an ImageNet-pretrained ResNet-34 encoder (He et al., 2016).\n"
            "2. **Ellipse fitting.** Each mask is resized to the original image, an ellipse is fitted "
            "to its outline, and the perimeter is computed with Ramanujan's approximation.\n"
            "3. **Bias correction and averaging.** A fixed pixel offset chosen on validation "
            "(7.19 px and 3.27 px) is removed from each model, and the two results are averaged "
            "(Dietterich, 2000).\n"
            "4. **Interpretation.** Gestational age from Hadlock et al. (1984). Percentiles from the "
            "INTERGROWTH-21st standard (Papageorghiou et al., 2014).")
        st.markdown(
            "**Quality checks.** A result is reported only when each mask overlaps its own fitted "
            "ellipse by at least 0.85 and the two masks agree with a Dice of at least 0.85.\n\n"
            "**Typical error.** Test set error stays close to constant in pixels (19.3 px under "
            "100 mm, 13.0 px from 100 to 200 mm, 15.6 px over 200 mm), so the range is scaled by "
            "each scan's own pixel size.")

    with st.expander("Validation"):
        st.markdown(
            "Held-out HC18 test set, 149 images:\n"
            "- Mean absolute error 1.94 mm (95% CI 1.66 to 2.26)\n"
            "- Bland-Altman bias +0.21 mm, limits −5.03 to +5.45 mm (Bland & Altman, 1986)\n"
            "- 95% within 5 mm of the expert, 62% within 2 mm\n"
            "- Mean age error 1.38 days\n"
            "- Second trimester error 1.76 mm, against 3.7 mm between two human observers "
            "(van den Heuvel et al., 2018)")

    with st.expander("References"):
        st.markdown(
            "- Bland, J. M., & Altman, D. G. (1986). *The Lancet, 327*(8476), 307–310.\n"
            "- Dietterich, T. G. (2000). Ensemble methods in machine learning. Springer.\n"
            "- Hadlock, F. P., et al. (1984). *Radiology, 152*(2), 497–501.\n"
            "- He, K., et al. (2016). Deep residual learning for image recognition. CVPR.\n"
            "- Papageorghiou, A. T., et al. (2014). *The Lancet, 384*(9946), 869–879.\n"
            "- Ronneberger, O., et al. (2015). U-Net. MICCAI 2015.\n"
            "- van den Heuvel, T. L. A., et al. (2018). *PLOS ONE, 13*(8), e0200412.")

    st.caption("Sample scans are from the HC18 dataset (van den Heuvel et al., 2018).")
    st.caption("AAI-521 Final Project, University of San Diego")

try:
    plain_model, resnet_model = load_models()
except Exception:
    st.error("The models could not be loaded. Check the network connection and refresh the page.")
    st.stop()
hc18_info = load_hc18_info()

samples = []
if os.path.isdir(SAMPLE_DIR):
    samples = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))

left, right = st.columns([3, 2], gap="large")

# pick an image
with left:
    upload = st.file_uploader("Ultrasound image", type=["png", "jpg", "jpeg"])
    pick = "None"
    if samples and upload is None:
        pick = st.selectbox("Or select a sample", ["None"] + samples)

if upload is not None:
    name = upload.name
    gray = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
elif pick != "None":
    name = pick
    gray = cv2.imread(os.path.join(SAMPLE_DIR, pick), cv2.IMREAD_GRAYSCALE)
else:
    st.info("Upload an image or select a sample to begin." if samples else "Upload an image to begin.")
    st.stop()

if gray is None:
    st.error("The image could not be read. Use a PNG or JPG file.")
    st.stop()

# HC18 images know their own pixel size and the expert HC
known = hc18_info.get(name)
with right:
    pixel_mm = st.number_input(
        "Pixel size (mm per pixel)", min_value=0.01, max_value=1.0, step=0.001, format="%.4f",
        value=float(known[0]) if known else 0.13,
        help="Physical size of one pixel. It is set by the scanner and changes with the zoom level.")
    if known:
        st.caption("Pixel size loaded from the HC18 dataset.")
    else:
        st.warning("This is not an HC18 file. Confirm the pixel size for this scan.")
    known_ga = st.number_input(
        "Gestational age from dating scan (weeks, optional)", min_value=14.0, max_value=40.0,
        value=None, step=0.1, format="%.1f",
        help="If the age is already known, the HC percentile is reported against the "
             "INTERGROWTH-21st standard.")

with st.spinner("Segmenting the fetal head..."):
    plain_mask = predict_mask(plain_model, gray, PLAIN_THRESHOLD, channels=1)
    resnet_mask = predict_mask(resnet_model, gray, RESNET_THRESHOLD, channels=3)

with left:
    st.image(draw_outlines(gray, plain_mask, resnet_mask), caption=name, width="stretch")

# fix the overestimate in pixels, then convert to mm
plain_hc = (hc_pixels(plain_mask) - PLAIN_OFFSET_PX) * pixel_mm
resnet_hc = (hc_pixels(resnet_mask) - RESNET_OFFSET_PX) * pixel_mm

plain_shape = head_shape_score(plain_mask)
resnet_shape = head_shape_score(resnet_mask)
agree = dice(plain_mask, resnet_mask)
passed = (plain_shape >= MIN_HEAD_SHAPE and resnet_shape >= MIN_HEAD_SHAPE
          and agree >= MIN_AGREE and not np.isnan(plain_hc) and not np.isnan(resnet_hc))

with right:
    st.subheader("Quality checks")
    st.markdown(checks_html([
        ("Head shape, plain U-Net", plain_shape, MIN_HEAD_SHAPE),
        ("Head shape, ResNet-34 U-Net", resnet_shape, MIN_HEAD_SHAPE),
        ("Model agreement (Dice)", agree, MIN_AGREE),
    ]), unsafe_allow_html=True)

    if not passed:
        st.error("Measurement withheld. The segmentations did not pass the quality checks.")
        st.write("This usually means the image is not the standard measurement plane, or the scan is "
                 "earlier than about 12 weeks, before the skull has ossified.")
        st.stop()

    final_hc = (plain_hc + resnet_hc) / 2
    error_mm = expected_error_mm(final_hc, pixel_mm)

    st.subheader("Results")
    st.metric("Head circumference", f"{final_hc:.1f} mm")
    st.caption(f"Typical error for this scan: ± {error_mm:.1f} mm, based on the average test set error in "
               "pixels at this pixel size and head size.")

    est_ga = None
    if final_hc >= 100:
        est_ga = hadlock_weeks(final_hc)
        low, high = hadlock_weeks(final_hc - error_mm), hadlock_weeks(final_hc + error_mm)
        days = (high - low) / 2 * 7
        st.metric("Estimated gestational age (Hadlock)", weeks_and_days(est_ga))
        st.caption(f"{trimester(est_ga)} trimester. Typical error: ± {days:.1f} days.")
    else:
        st.write("No age estimate. Hadlock's formula applies to heads of 100 mm and above.")

    if known:
        expert_hc = known[1]
        diff = final_hc - expert_hc
        note = f"Expert annotation: {expert_hc:.1f} mm. Difference: {diff:+.1f} mm"
        if expert_hc >= 100 and final_hc >= 100:
            day_diff = (hadlock_weeks(final_hc) - hadlock_weeks(expert_hc)) * 7
            note += f" ({day_diff:+.1f} days)"
        st.write(note + ".")
        if known[2] != "test":
            if known[2] == "train":
                st.caption("This scan is from the training set, so the models learned from it. "
                           "Use a test set scan for an unbiased comparison.")
            else:
                st.caption("This scan is from the validation set, which was used to choose the "
                           "threshold and bias correction. Use a test set scan for an unbiased comparison.")

    with st.expander("Individual model measurements"):
        st.write(f"ResNet-34 U-Net: {resnet_hc:.1f} mm")
        st.write(f"Plain U-Net: {plain_hc:.1f} mm")

with left:
    st.subheader("Growth chart (INTERGROWTH-21st)")
    if known_ga is not None:
        pct, z = centile(final_hc, known_ga)
        if pct < 1:
            where = "below the **1st percentile**"
        elif pct > 99:
            where = "above the **99th percentile**"
        else:
            where = f"at the **{ordinal(pct)} percentile**"
        st.write(f"At {weeks_and_days(known_ga)}, this HC is {where} (z = {z:+.2f}).")
        if pct < 3 or pct > 97:
            st.warning("This HC is outside the 3rd to 97th percentile range for the given age.")
        st.altair_chart(growth_chart(final_hc, known_ga, "Measured HC at known age"), width="stretch")
    elif est_ga is not None and 14 <= est_ga <= 40:
        st.write("The point is placed at the estimated age. Enter the gestational age from a dating "
                 "scan to report a percentile.")
        st.altair_chart(growth_chart(final_hc, est_ga, "Measured HC at estimated age"), width="stretch")
    else:
        st.write("This head is outside the 14 to 40 week range of the standard.")
        st.altair_chart(growth_chart(final_hc, None, ""), width="stretch")
    st.caption("Shaded band: 3rd to 97th percentile. Solid line: 50th percentile "
               "(Papageorghiou et al., 2014).")
