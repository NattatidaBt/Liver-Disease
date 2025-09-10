import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
import os
from datetime import datetime
import pandas as pd

# =========================
# CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "3mobilenetv3_large_100_checkpoint_fold1.pt")

MODEL_NAME = "mobilenetv3_large_100"
NUM_CLASSES = 5
CLASS_NAMES = ["F1_F2", "F3", "F4", "Mass", "Normal"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    """Loads the pre-trained MobileNetV3 model."""
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    return model


model = load_model()


# =========================
# Preprocess Function
# =========================
def preprocess(img_pil):
    """Preprocesses a PIL image for model inference."""
    img = img_pil.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = arr.transpose(2, 0, 1)
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return tensor


# =========================
# Session State Management
# =========================
if "results" not in st.session_state:
    st.session_state["results"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []
if "last_uploaded" not in st.session_state:
    st.session_state["last_uploaded"] = None
if "carousel_index" not in st.session_state:
    st.session_state["carousel_index"] = 0
if "clear_history_clicked" not in st.session_state:
    st.session_state["clear_history_clicked"] = False

# =========================
# Streamlit UI Layout
# =========================
st.title("Multi-Class Liver Disease Classification in Ultrasound Images Using CNN")

uploaded = st.file_uploader(
    "Upload an ultrasound image of the liver",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Process uploaded image
# =========================
# Check if clear button was clicked to prevent re-processing the same file
if uploaded is not None and uploaded != st.session_state["last_uploaded"] and not st.session_state[
    "clear_history_clicked"]:
    img_pil = Image.open(uploaded).convert("RGB")
    img_display = img_pil.resize((499, 499))

    tensor = preprocess(img_pil)
    with torch.no_grad():
        preds = model(tensor)
        probs = torch.softmax(preds, dim=1)[0].cpu().numpy()
    pred_idx = np.argmax(probs)
    pred_label = CLASS_NAMES[pred_idx]

    file_name = uploaded.name
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add new result and history entry
    st.session_state["results"].append({
        "image": img_display,
        "probs": probs,
        "pred_label": pred_label,
        "file_name": file_name,
        "upload_time": upload_time
    })

    st.session_state["history"].append({
        "file_name": file_name,
        "upload_time": upload_time,
        "pred_label": pred_label,
        **{cls: float(probs[idx]) for idx, cls in enumerate(CLASS_NAMES)}
    })

    st.session_state["last_uploaded"] = uploaded
    st.session_state["carousel_index"] = max(len(st.session_state["results"]) - 4, 0)
    st.rerun()

# ---
# Uploaded Images & Predictions (Carousel)
# ---
images_per_page = 4
if st.session_state["results"]:
    st.subheader("Uploaded Images & Predictions (Carousel)")

    total_images = len(st.session_state["results"])
    idx = st.session_state["carousel_index"]
    max_index = max(total_images - images_per_page, 0)

    col_prev, col_display, col_next = st.columns([1, 8, 1])

    with col_prev:
        st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
        if idx > 0:
            if st.button("⬅", key="prev_btn"):
                st.session_state["carousel_index"] = max(0, idx - images_per_page)
                st.rerun()

    with col_display:
        display_results = st.session_state["results"][idx:idx + images_per_page]
        cols = st.columns(images_per_page)
        for i, res in enumerate(display_results):
            with cols[i]:
                st.image(res["image"], caption=f"{res['file_name']} ({res['upload_time']})", use_container_width=True)

                st.markdown("**Prediction Result:**")
                for j, cls in enumerate(CLASS_NAMES):
                    prob = res["probs"][j] * 100
                    if cls == res["pred_label"]:
                        st.markdown(f"**<span style='color:blue;'>{cls}: {prob:.2f}%</span>**", unsafe_allow_html=True)
                    else:
                        st.markdown(f"{cls}: {prob:.2f}%")

    with col_next:
        st.markdown("<div style='height: 120px;'></div>", unsafe_allow_html=True)
        if idx + images_per_page < total_images:
            if st.button("➡", key="next_btn"):
                st.session_state["carousel_index"] = min(idx + images_per_page, max_index)
                st.rerun()

    current_page = idx // images_per_page + 1
    total_pages = (total_images + images_per_page - 1) // images_per_page
    st.markdown(f"<div style='text-align:center;'>Page {current_page} / {total_pages}</div>", unsafe_allow_html=True)

# ---
# Prediction History & Data Management
# ---
if st.session_state["history"]:
    st.subheader("Prediction History")
    df = pd.DataFrame(st.session_state["history"])

    # Format the numeric columns to 2 decimal places
    for cls in CLASS_NAMES:
        df[cls] = df[cls].apply(lambda x: f"{x:.2f}")

    # Reset index to start from 1
    df.index = df.index + 1
    st.dataframe(df, use_container_width=True)

    row_to_delete = st.selectbox(
        "Select row to delete from history:",
        options=["None"] + list(df.index)
    )
    if row_to_delete != "None":
        if st.button("Delete Selected Row"):
            deleted_index = int(row_to_delete) - 1
            deleted_file_name = st.session_state["history"][deleted_index]["file_name"]
            st.session_state["history"].pop(deleted_index)
            st.session_state["results"] = [r for r in st.session_state["results"] if
                                           r["file_name"] != deleted_file_name]
            # Set the flag to prevent re-processing after this action
            st.session_state["clear_history_clicked"] = True
            st.rerun()

    col_clear, col_download = st.columns([1, 1])
    with col_clear:
        if st.button("Clear All History"):
            st.session_state["history"] = []
            st.session_state["results"] = []
            st.session_state["carousel_index"] = 0
            st.session_state["last_uploaded"] = None
            # Set a flag to prevent re-processing the last uploaded file
            st.session_state["clear_history_clicked"] = True
            st.rerun()

    with col_download:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )