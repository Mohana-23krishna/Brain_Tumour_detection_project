import streamlit as st
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
import random
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")
 
# ── CSS + Hero ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');
 
:root {
  --bg:      #04080f;
  --surface: #0b1220;
  --border:  #1a2a40;
  --cyan:    #00e5ff;
  --cdim:    #007a99;
  --white:   #e8f4f8;
  --muted:   #6a8a9a;
  --red:     #ff4d6d;
  --green:   #00e676;
  --amber:   #ffab40;
}
 
/* Base */
.stApp { background: var(--bg); color: var(--white); font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0; padding-bottom: 3rem; max-width: 1180px; }
 
/* ─ Hero ─ */
.hero-wrap {
  position: relative; overflow: hidden;
  display: flex; flex-direction: column; align-items: center;
  padding: 3.2rem 1rem 2.4rem; text-align: center;
}
.hero-glow {
  position: absolute; width: 700px; height: 700px; border-radius: 50%;
  background: radial-gradient(circle, rgba(0,229,255,.09) 0%, transparent 68%);
  top: 50%; left: 50%; transform: translate(-50%,-50%); pointer-events: none;
}
.pulse-ring-wrap {
  position: relative; width: 88px; height: 88px;
  display: flex; align-items: center; justify-content: center; margin-bottom: 1.4rem;
}
.pulse-ring-wrap::before, .pulse-ring-wrap::after {
  content: ''; position: absolute; border-radius: 50%;
  border: 1.5px solid rgba(0,229,255,.7);
  animation: ring 2.6s ease-out infinite;
}
.pulse-ring-wrap::before { width: 100%; height: 100%; }
.pulse-ring-wrap::after  { width: 155%; height: 155%; animation-delay: .9s; opacity:.45; }
@keyframes ring {
  0%   { transform: scale(.8);  opacity: .9; }
  100% { transform: scale(1.7); opacity: 0;  }
}
.brain { font-size: 3rem; filter: drop-shadow(0 0 16px var(--cyan)); }
 
.hero-title {
  font-family: 'Orbitron', sans-serif; font-weight: 900;
  font-size: clamp(2.2rem, 5vw, 3.6rem);
  letter-spacing: .14em; line-height: 1.1; margin: 0 0 .35rem;
  color: var(--white);
}
.hero-title .accent {
  color: var(--cyan);
  text-shadow: 0 0 18px var(--cyan), 0 0 50px rgba(0,229,255,.28);
}
.hero-sub {
  font-weight: 300; font-size: .95rem; letter-spacing: .22em;
  text-transform: uppercase; color: var(--muted); margin: 0 0 1.8rem;
}
.badges { display: flex; gap: .55rem; flex-wrap: wrap; justify-content: center; }
.badge {
  background: rgba(0,229,255,.07); border: 1px solid rgba(0,229,255,.22);
  border-radius: 20px; padding: .26rem .9rem;
  font-size: .7rem; letter-spacing: .08em; color: var(--cyan); font-weight: 500;
}
.divider {
  width: 100%; height: 1px; margin: 0 0 1.6rem;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
}
 
/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {
  gap: 0; background: var(--surface);
  border-radius: 10px; padding: 4px; border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  border-radius: 7px; color: var(--muted); background: transparent; border: none;
  font-family: 'DM Sans', sans-serif; font-weight: 500;
  font-size: .87rem; letter-spacing: .04em; padding: .48rem 1.2rem;
}
.stTabs [aria-selected="true"] {
  background: rgba(0,229,255,.12) !important;
  color: var(--cyan) !important;
  box-shadow: inset 0 0 0 1px rgba(0,229,255,.3);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }
 
/* ─ Cards ─ */
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;
  position: relative; overflow: hidden;
}
.card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
}
.card-label {
  font-family: 'Orbitron', sans-serif; font-size: .62rem;
  letter-spacing: .18em; color: var(--cyan); text-transform: uppercase; margin-bottom: .7rem;
}
 
/* ─ Section title ─ */
.sec {
  font-family: 'Orbitron', sans-serif; font-size: .88rem; font-weight: 700;
  color: var(--white); letter-spacing: .1em;
  margin: 1.4rem 0 .8rem;
  display: flex; align-items: center; gap: .6rem;
}
.sec::after { content: ''; flex: 1; height: 1px; background: var(--border); }
 
/* ─ Result boxes ─ */
.rbox {
  border-radius: 11px; padding: 1rem 1.4rem;
  margin-top: .8rem; border-left: 4px solid;
  font-size: .95rem; font-weight: 500;
}
.r-green  { background: rgba(0,230,118,.08);  border-color: var(--green);  color: var(--green); }
.r-red    { background: rgba(255,77,109,.08); border-color: var(--red);    color: var(--red); }
.r-amber  { background: rgba(255,171,64,.08); border-color: var(--amber);  color: var(--amber); }
.r-cyan   { background: rgba(0,229,255,.07);  border-color: var(--cyan);   color: var(--cyan); }
 
/* ─ Empty state ─ */
.empty-state {
  min-height: 260px; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: .8rem;
  border: 1px dashed var(--border); border-radius: 14px;
  background: var(--surface);
}
 
/* ─ Inputs ─ */
.stTextInput input, .stNumberInput input {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--white) !important;
  font-family: 'DM Sans', sans-serif !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px rgba(0,229,255,.15) !important;
}
label { color: var(--muted) !important; font-size: .82rem !important; letter-spacing: .05em; }
 
/* ─ Buttons ─ */
div.stButton > button {
  background: linear-gradient(135deg, #006680, #00e5ff);
  color: #000 !important; font-family: 'Orbitron', sans-serif;
  font-size: .74rem; font-weight: 700; letter-spacing: .1em;
  border: none; border-radius: 8px; padding: .6rem 2rem;
  text-transform: uppercase; transition: all .2s;
}
div.stButton > button:hover {
  transform: translateY(-1px); box-shadow: 0 6px 22px rgba(0,229,255,.28);
}
 
/* ─ File uploader ─ */
[data-testid="stFileUploader"] {
  background: var(--surface); border: 1px dashed var(--cdim); border-radius: 12px; padding: 1rem;
}
[data-testid="stFileUploader"]:hover { border-color: var(--cyan); }
 
/* ─ Progress ─ */
.stProgress > div > div {
  background: linear-gradient(90deg, #006680, #00e5ff) !important; border-radius: 4px;
}
 
/* ─ Image ─ */
.stImage img { border: 1px solid var(--border); border-radius: 10px; }
 
/* ─ Spinner ─ */
.stSpinner > div { border-top-color: var(--cyan) !important; }
</style>
 
<div class="hero-wrap">
  <div class="hero-glow"></div>
  <div class="pulse-ring-wrap"><span class="brain">🧠</span></div>
  <h1 class="hero-title">NEURO<span class="accent">SCAN</span> AI</h1>
  <p class="hero-sub">Brain Tumor Detection &amp; Classification System</p>
  <div class="badges">
    <span class="badge">⚡ VGG16 Architecture</span>
    <span class="badge">🎯 4-Class Detection</span>
    <span class="badge">🔬 MRI Analysis</span>
    <span class="badge">📊 Real-time Inference</span>
  </div>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)
 
# ── Constants ─────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 128
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']
TRAIN_DIR    = "MRI_Images/Training"
TEST_DIR     = "MRI_Images/Testing"
 
# ── Dark matplotlib helper ────────────────────────────────────────────────────
def dark_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    for obj in (fig, ax):
        try: obj.set_facecolor('#0b1220')
        except: pass
    ax.tick_params(colors='#6a8a9a', labelsize=8)
    ax.xaxis.label.set_color('#6a8a9a')
    ax.yaxis.label.set_color('#6a8a9a')
    ax.title.set_color('#e8f4f8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a2a40')
    return fig, ax
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def augment_image(image):
    image = Image.fromarray(np.uint8(image))
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
    return np.array(image) / 255.0
 
def open_images(paths, augment=False):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    out = []
    for p in paths:
        arr = img_to_array(load_img(p, target_size=(IMAGE_SIZE, IMAGE_SIZE)))
        out.append(augment_image(arr) if augment else arr / 255.0)
    return np.array(out)
 
def load_dataset_paths(data_dir):
    paths, labels = [], []
    if not os.path.exists(data_dir):
        return paths, labels
    for cls in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                paths.append(os.path.join(d, f)); labels.append(cls)
    return paths, labels
 
def encode_label(labels, c2i):
    return np.array([c2i[l] for l in labels])
 
def datagen(paths, labels, c2i, batch_size=12):
    while True:
        for i in range(0, len(paths), batch_size):
            yield open_images(paths[i:i+batch_size], augment=True), encode_label(labels[i:i+batch_size], c2i)
 
@st.cache_resource
def build_model(num_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.optimizers import Adam
    base = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    for layer in base.layers: layer.trainable = False
    for layer in base.layers[-4:-1]: layer.trainable = True
    m = Sequential([Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)), base,
                    Flatten(), Dropout(0.3), Dense(128, activation='relu'),
                    Dropout(0.2), Dense(num_classes, activation='softmax')])
    m.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
    return m

# ── FIX: Robust model loader — 3-attempt strategy ────────────────────────────
# Attempt 1 → clean load (perfect version match)
# Attempt 2 → patch InputLayer config (handles missing/wrong kwargs)
# Attempt 3 → rebuild architecture + load weights only (bypasses all config)
@st.cache_resource
def load_saved_model(path):
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    # ── Attempt 1: clean load ────────────────────────────────────────────────
    try:
        return load_model(path, compile=False)
    except Exception:
        pass

    # ── Attempt 2: patch InputLayer config ───────────────────────────────────
    try:
        from tensorflow.keras import layers as kl
        original_from_config = kl.InputLayer.from_config

        @classmethod
        def patched_from_config(cls, config):
            config.pop("batch_shape", None)
            config.pop("optional", None)
            # Inject shape if missing entirely
            if "shape" not in config:
                config["shape"] = (IMAGE_SIZE, IMAGE_SIZE, 3)
            return original_from_config.__func__(cls, config)

        kl.InputLayer.from_config = patched_from_config
        try:
            model = load_model(path, compile=False)
            return model
        finally:
            kl.InputLayer.from_config = original_from_config
    except Exception:
        pass

    # ── Attempt 3: rebuild architecture + load weights only ──────────────────
    # Completely bypasses config deserialization — works regardless of how
    # the .h5 was saved, as long as the architecture hasn't changed.
    try:
        model = build_model(len(CLASS_LABELS))
        model.load_weights(path)
        return model
    except Exception as final_err:
        raise RuntimeError(
            f"Could not load model after three attempts.\n\n"
            f"Root cause: {final_err}\n\n"
            f"Deployment is running TF {tf.__version__}. The most reliable fix is "
            f"to re-save your model.h5 locally using TF 2.16.1:\n"
            f"  model.save('model.h5')"
        )

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["🔍  Detect", "🏋️  Train", "📊  Evaluate", "🖼️  Dataset"])
 
# ─────────────────────────────── DETECT ──────────────────────────────────────
with t1:
    st.markdown('<div class="sec">🔍 MRI Scan Analysis</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")
 
    with left:
        st.markdown('<div class="card"><div class="card-label">Configuration</div>', unsafe_allow_html=True)
        model_path = st.text_input("Model path", value="model.h5")
        uploaded   = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded:
            st.markdown('<div class="card"><div class="card-label">Input Scan</div>', unsafe_allow_html=True)
            st.image(uploaded, width=270)
            st.markdown('</div>', unsafe_allow_html=True)
 
    with right:
        if uploaded:
            if not os.path.exists(model_path):
                st.markdown(
                    f'<div class="rbox r-red">⚠ Model <code>{model_path}</code> not found. Train first.</div>',
                    unsafe_allow_html=True)
            else:
                try:
                    from tensorflow.keras.preprocessing.image import load_img, img_to_array
                    mdl   = load_saved_model(model_path)
                    arr   = np.expand_dims(
                                img_to_array(load_img(uploaded, target_size=(IMAGE_SIZE, IMAGE_SIZE))) / 255., 0)
                    preds = mdl.predict(arr)
                    idx   = int(np.argmax(preds))
                    conf  = float(np.max(preds))
                    label = CLASS_LABELS[idx]
 
                    st.markdown('<div class="card"><div class="card-label">Confidence Scores</div>',
                                unsafe_allow_html=True)
                    fig, ax = dark_fig(5, 2.8)
                    colors = ['#00e5ff'] * 4
                    colors[idx] = '#00e676' if label == 'notumor' else '#ff4d6d'
                    bars = ax.barh(CLASS_LABELS, preds[0], color=colors, height=0.5)
                    ax.set_xlim(0, 1); ax.set_xlabel("Confidence", fontsize=9)
                    for bar, v in zip(bars, preds[0]):
                        ax.text(v + .01, bar.get_y() + bar.get_height() / 2,
                                f'{v * 100:.1f}%', va='center', color='#6a8a9a', fontsize=8)
                    plt.tight_layout(); st.pyplot(fig); plt.close()
                    st.markdown('</div>', unsafe_allow_html=True)
 
                    if label == "notumor":
                        st.markdown(
                            f'<div class="rbox r-green">✅ No Tumor Detected &nbsp;·&nbsp; Confidence: {conf*100:.1f}%</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            '<div class="rbox r-cyan">💡 You\'re clear! Maintain healthy habits and regular check-ups.</div>',
                            unsafe_allow_html=True)
                    else:
                        box_cls = "r-amber" if conf < 0.6 else "r-red"
                        tag     = "⚠️ Low Confidence" if conf < 0.6 else "🔴 High Confidence"
                        st.markdown(
                            f'<div class="rbox {box_cls}">{tag} &nbsp;·&nbsp; Tumor: <strong>{label.upper()}</strong> &nbsp;·&nbsp; {conf*100:.1f}%</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            '<div class="rbox r-amber">⚕️ Consult a neurologist immediately for proper diagnosis.</div>',
                            unsafe_allow_html=True)
                        st.markdown(
                            '<div class="rbox r-cyan">💡 Healthy diet &nbsp;·&nbsp; Reduce stress &nbsp;·&nbsp; Follow doctor\'s advice &nbsp;·&nbsp; No self-medication</div>',
                            unsafe_allow_html=True)
 
                except Exception as e:
                    st.markdown(
                        f'<div class="rbox r-red">❌ Model load failed: <code>{str(e)}</code></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="empty-state">'
                '<span style="font-size:2.8rem;opacity:.25">🧠</span>'
                '<p style="color:#6a8a9a;font-size:.9rem;margin:0">Upload an MRI scan to begin analysis</p>'
                '</div>',
                unsafe_allow_html=True)
 
# ─────────────────────────────── TRAIN ───────────────────────────────────────
with t2:
    st.markdown('<div class="sec">🏋️ Model Training</div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-label">Hyperparameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    epochs     = c1.number_input("Epochs",      1, 50, 5)
    batch_size = c2.number_input("Batch size",  4, 64, 20)
    save_path  = c3.text_input("Save model as", "model.h5")
    st.markdown('</div>', unsafe_allow_html=True)
 
    if st.button("🚀  Start Training"):
        train_paths, train_labels = load_dataset_paths(TRAIN_DIR)
        if not train_paths:
            st.markdown(
                f'<div class="rbox r-red">No data found in <code>{TRAIN_DIR}</code>. Check folder structure.</div>',
                unsafe_allow_html=True)
        else:
            class_names = sorted(set(train_labels))
            c2i = {n: i for i, n in enumerate(class_names)}
            st.markdown(
                f'<div class="rbox r-cyan">📂 {len(train_paths)} images &nbsp;·&nbsp; {len(class_names)} classes: {class_names}</div>',
                unsafe_allow_html=True)
            mdl    = build_model(len(class_names))
            steps  = max(1, len(train_paths) // batch_size)
            bar    = st.progress(0)
            status = st.empty()
            acc_h, loss_h = [], []
            for ep in range(epochs):
                h = mdl.fit(datagen(train_paths, train_labels, c2i, batch_size),
                            steps_per_epoch=steps, epochs=1, verbose=0)
                acc_h.append(h.history['sparse_categorical_accuracy'][0])
                loss_h.append(h.history['loss'][0])
                bar.progress((ep + 1) / epochs)
                status.markdown(
                    f'<div class="rbox r-cyan">Epoch {ep+1}/{epochs} &nbsp;·&nbsp; '
                    f'Acc: {acc_h[-1]:.4f} &nbsp;·&nbsp; Loss: {loss_h[-1]:.4f}</div>',
                    unsafe_allow_html=True)
            mdl.save(save_path)
            st.markdown(
                f'<div class="rbox r-green">✅ Model saved → <code>{save_path}</code></div>',
                unsafe_allow_html=True)
            fig, ax = dark_fig(8, 4)
            ax.plot(acc_h,  '.-', color='#00e676', lw=2, label='Accuracy')
            ax.plot(loss_h, '.-', color='#ff4d6d', lw=2, label='Loss')
            ax.set_xlabel('Epoch'); ax.set_title('Training History')
            ax.legend(facecolor='#0b1220', labelcolor='#e8f4f8')
            plt.tight_layout(); st.pyplot(fig); plt.close()
 
# ─────────────────────────────── EVALUATE ────────────────────────────────────
with t3:
    st.markdown('<div class="sec">📊 Model Evaluation</div>', unsafe_allow_html=True)
    eval_path = st.text_input("Model path for evaluation", "model.h5", key="eval")
    if st.button("📈  Run Evaluation"):
        if not os.path.exists(eval_path):
            st.markdown(
                f'<div class="rbox r-red">Model not found: <code>{eval_path}</code></div>',
                unsafe_allow_html=True)
        else:
            test_paths, test_raw = load_dataset_paths(TEST_DIR)
            if not test_paths:
                st.markdown(
                    f'<div class="rbox r-red">No test data in <code>{TEST_DIR}</code></div>',
                    unsafe_allow_html=True)
            else:
                try:
                    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
                    from sklearn.preprocessing import label_binarize
                    import pandas as pd
 
                    mdl = load_saved_model(eval_path)
                    cns = sorted(set(test_raw))
                    c2i = {n: i for i, n in enumerate(cns)}
 
                    with st.spinner("Running predictions…"):
                        imgs = open_images(test_paths)
                        enc  = encode_label(test_raw, c2i)
                        pred = mdl.predict(imgs)
                        pc   = np.argmax(pred, axis=1)
 
                    st.markdown('<div class="sec">Classification Report</div>', unsafe_allow_html=True)
                    df = pd.DataFrame(
                            classification_report(enc, pc, target_names=cns, output_dict=True)).T
                    st.dataframe(df.style.background_gradient(cmap='Blues'))
 
                    ca, cb = st.columns(2, gap="medium")
                    with ca:
                        st.markdown('<div class="sec">Confusion Matrix</div>', unsafe_allow_html=True)
                        cm = confusion_matrix(enc, pc)
                        fig, ax = dark_fig(5, 4)
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=cns, yticklabels=cns, ax=ax,
                                    linewidths=.5, linecolor='#1a2a40')
                        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
                        ax.set_title('Confusion Matrix')
                        plt.tight_layout(); st.pyplot(fig); plt.close()
 
                    with cb:
                        st.markdown('<div class="sec">ROC Curves</div>', unsafe_allow_html=True)
                        lb   = label_binarize(enc, classes=np.arange(len(cns)))
                        rocs = ['#00e5ff', '#00e676', '#ffab40', '#ff4d6d']
                        fig, ax = dark_fig(5, 4)
                        for i, cls in enumerate(cns):
                            fp, tp, _ = roc_curve(lb[:, i], pred[:, i])
                            ax.plot(fp, tp, color=rocs[i % 4],
                                    label=f'{cls} (AUC={auc(fp,tp):.2f})', lw=1.8)
                        ax.plot([0, 1], [0, 1], '--', color='#1a2a40')
                        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
                        ax.set_title('ROC Curve')
                        ax.legend(facecolor='#0b1220', labelcolor='#e8f4f8', fontsize=8)
                        plt.tight_layout(); st.pyplot(fig); plt.close()
 
                except Exception as e:
                    st.markdown(
                        f'<div class="rbox r-red">❌ Evaluation failed: <code>{str(e)}</code></div>',
                        unsafe_allow_html=True)
 
# ─────────────────────────────── DATASET ─────────────────────────────────────
with t4:
    st.markdown('<div class="sec">🖼️ Dataset Explorer</div>', unsafe_allow_html=True)
    split    = st.selectbox("Split", ["Training", "Testing"])
    data_dir = TRAIN_DIR if split == "Training" else TEST_DIR
    paths, labels = load_dataset_paths(data_dir)
 
    if not paths:
        st.markdown(
            f'<div class="rbox r-amber">No data found in <code>{data_dir}</code></div>',
            unsafe_allow_html=True)
    else:
        from collections import Counter
        counts = Counter(labels)
        d1, d2 = st.columns([1, 1], gap="large")
 
        with d1:
            st.markdown('<div class="sec">Class Distribution</div>', unsafe_allow_html=True)
            pal  = ['#00e5ff', '#00e676', '#ffab40', '#ff4d6d']
            fig, ax = dark_fig(5, 3)
            bars = ax.bar(counts.keys(), counts.values(), color=pal[:len(counts)], width=0.5)
            for b, v in zip(bars, counts.values()):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                        str(v), ha='center', color='#e8f4f8', fontsize=9)
            ax.set_ylabel('Count'); ax.set_title(f'{split} Set')
            plt.tight_layout(); st.pyplot(fig); plt.close()
 
        with d2:
            st.markdown('<div class="card"><div class="card-label">Dataset Stats</div>', unsafe_allow_html=True)
            total = len(paths)
            for i, (cls, cnt) in enumerate(counts.items()):
                pct   = cnt / total * 100
                color = pal[i % len(pal)]
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                     padding:.45rem 0;border-bottom:1px solid #1a2a40;font-size:.88rem;">
                  <span style="color:#e8f4f8">{cls}</span>
                  <span style="color:{color};font-family:'Orbitron',monospace;font-size:.78rem">
                    {cnt} <span style="color:#6a8a9a">({pct:.1f}%)</span>
                  </span>
                </div>""", unsafe_allow_html=True)
            st.markdown(
                f'<div style="margin-top:.9rem;color:#6a8a9a;font-size:.8rem">Total: {total} images</div>',
                unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
 
        st.markdown('<div class="sec">Random Sample</div>', unsafe_allow_html=True)
        sample_p = random.sample(paths, min(10, len(paths)))
        sample_l = [labels[paths.index(p)] for p in sample_p]
        lc = {'glioma': '#ff4d6d', 'meningioma': '#ffab40', 'notumor': '#00e676', 'pituitary': '#00e5ff'}
 
        fig, axes = plt.subplots(2, 5, figsize=(16, 6.5))
        fig.patch.set_facecolor('#0b1220')
        for i, (p, lbl) in enumerate(zip(sample_p, sample_l)):
            ax = axes.ravel()[i]
            ax.imshow(Image.open(p).resize((IMAGE_SIZE, IMAGE_SIZE)))
            ax.axis("off")
            c = lc.get(lbl, '#e8f4f8')
            ax.set_title(lbl, color=c, fontsize=10, fontweight='bold', pad=5)
            for spine in ax.spines.values():
                spine.set_edgecolor(c); spine.set_linewidth(1.5)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()
