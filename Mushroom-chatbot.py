"""
🍄 Mushroom Safety Chatbot — Streamlit App
==========================================
A friendly conversational chatbot that predicts whether a mushroom is
edible or poisonous using an XGBoost model trained on the Mushroom dataset.

HOW TO RUN
----------
1. Install dependencies (one-time):
   pip install streamlit xgboost scikit-learn pandas numpy

2. Place "Mushroom_data.csv" in the same folder as this file.

3. Launch:
   streamlit run mushroom_chatbot.py
"""

import random
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍄 Mushroom Safety Bot",
    page_icon="🍄",
    layout="centered",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);}
.user-bubble{background:linear-gradient(135deg,#667eea,#764ba2);color:white;
  border-radius:18px 18px 4px 18px;padding:12px 16px;margin:6px 0 6px 60px;
  box-shadow:0 4px 12px rgba(102,126,234,.4);font-size:15px;}
.bot-bubble{background:linear-gradient(135deg,#1e3a5f,#2d5a8e);color:#e8f4f8;
  border-radius:18px 18px 18px 4px;padding:12px 16px;margin:6px 60px 6px 0;
  border:1px solid rgba(100,180,255,.2);box-shadow:0 4px 12px rgba(0,0,0,.3);font-size:15px;}
.avatar{font-size:28px;margin-bottom:2px;}
.edible-card{background:linear-gradient(135deg,#0f5132,#198754);color:white;
  border-radius:16px;padding:20px;text-align:center;font-size:22px;font-weight:bold;
  box-shadow:0 8px 24px rgba(25,135,84,.5);margin:10px 0;}
.poisonous-card{background:linear-gradient(135deg,#6a0000,#dc3545);color:white;
  border-radius:16px;padding:20px;text-align:center;font-size:22px;font-weight:bold;
  box-shadow:0 8px 24px rgba(220,53,69,.5);margin:10px 0;}
.confidence-bar{background:rgba(255,255,255,.2);border-radius:10px;height:10px;margin-top:8px;}
.confidence-fill{background:white;border-radius:10px;height:10px;}
.step-indicator{background:rgba(255,255,255,.07);border-radius:10px;padding:10px 14px;
  margin-bottom:8px;color:#a0c4ff;font-size:13px;}
.main-header{text-align:center;background:linear-gradient(135deg,rgba(102,126,234,.15),rgba(118,75,162,.15));
  border-radius:20px;padding:24px 16px 16px;border:1px solid rgba(100,180,255,.15);margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Feature metadata
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_META = {
    "cap-shape": {"options": {
        "bell":"b","conical":"c","convex":"x","flat":"f","sunken":"s","spherical":"p","others":"o"}},
    "cap-surface": {"options": {
        "fibrous":"i","grooves":"g","scaly":"y","smooth":"s","shiny":"h",
        "leathery":"l","silky":"k","sticky":"t","wrinkled":"w","fleshy":"e"}},
    "cap-color": {"options": {
        "brown":"n","buff":"b","gray":"g","green":"r","pink":"p","purple":"u",
        "red":"e","white":"w","yellow":"y","blue":"l","orange":"o","black":"k"}},
    "does-bruise-or-bleed": {"options": {"yes (bruises/bleeds)":"t","no":"f"}},
    "gill-attachment": {"options": {
        "adnate":"a","adnexed":"x","decurrent":"d","free":"e","sinuate":"s","pores":"p","none":"f"}},
    "gill-color": {"options": {
        "brown":"n","buff":"b","gray":"g","green":"r","pink":"p","purple":"u",
        "red":"e","white":"w","yellow":"y","blue":"l","orange":"o","black":"k","none":"f"}},
    "stem-color": {"options": {
        "brown":"n","buff":"b","gray":"g","green":"r","pink":"p","purple":"u",
        "red":"e","white":"w","yellow":"y","blue":"l","orange":"o","black":"k","none":"f"}},
    "has-ring": {"options": {"yes (has ring)":"t","no ring":"f"}},
    "ring-type": {"options": {
        "cobwebby":"c","evanescent":"e","flaring":"r","grooved":"g","large":"l",
        "pendant":"p","sheathing":"s","zone":"z","scaly":"y","movable":"m","none":"f"}},
    "habitat": {"options": {
        "grasses":"g","leaves":"l","meadows":"m","paths":"p",
        "heaths":"h","urban":"u","waste":"w","woods":"d"}},
    "season": {"options": {"spring":"s","summer":"u","autumn":"a","winter":"w"}},
}

NUMERIC_FEATURES = ["cap-diameter", "stem-height", "stem-width"]

QUESTION_POOL = {
    "cap-diameter": [
        "📏 How wide is the mushroom cap? (enter a number in cm, e.g. 5.0)",
        "📐 What's the cap diameter in centimetres?",
        "🔢 Can you estimate the cap width for me? (in cm)",
    ],
    "cap-shape": [
        "🍄 What shape is the mushroom cap?",
        "👀 Looking at it from the side — how would you describe the cap shape?",
        "🔍 What does the overall cap shape look like to you?",
    ],
    "cap-surface": [
        "✋ What does the cap surface feel and look like?",
        "🖐️ If you run a finger across the top, what's the texture like?",
        "🌿 How would you describe the surface of the cap?",
    ],
    "cap-color": [
        "🎨 What colour is the cap?",
        "🌈 What's the main colour on top of the mushroom?",
        "👁️ What colour would you say the cap is?",
    ],
    "does-bruise-or-bleed": [
        "💧 Does it bruise or bleed when you cut or press it?",
        "🔪 If you nick the flesh lightly, does it change colour or release liquid?",
        "🩸 Does the mushroom bruise or ooze anything when damaged?",
    ],
    "gill-attachment": [
        "🔗 How are the gills attached to the stem?",
        "👇 Flip it over — how do the gills connect to the stem?",
        "🧐 What's the gill-to-stem attachment like?",
    ],
    "gill-color": [
        "🎨 What colour are the gills underneath the cap?",
        "🌿 Peek under the cap — what colour are the gills?",
        "👇 Look at the underside — what's the gill colour?",
    ],
    "stem-height": [
        "📏 How tall is the stem? (in cm)",
        "📐 What's the stem height in centimetres?",
        "🔢 Give me the stem length in cm.",
    ],
    "stem-width": [
        "📏 How thick is the stem? (in mm)",
        "📐 What's the stem width in millimetres?",
        "🔢 How wide is the stem in mm?",
    ],
    "stem-color": [
        "🎨 What colour is the stem / stalk?",
        "🌿 What colour is the stalk?",
        "👀 What's the stem colour?",
    ],
    "has-ring": [
        "💍 Does the stem have a ring (skirt) on it?",
        "🔍 Is there a ring or skirt-like structure on the stem?",
        "👀 Do you see a ring on the stem?",
    ],
    "ring-type": [
        "💍 What type of ring is on the stem?",
        "🔍 Can you describe the ring — its shape or style?",
        "👁️ What does the ring look like?",
    ],
    "habitat": [
        "🌳 Where did you find this mushroom?",
        "📍 What kind of environment was it growing in?",
        "🗺️ What's the habitat like where you spotted it?",
    ],
    "season": [
        "🗓️ What season is it where you found this?",
        "📅 Which season did you find this mushroom in?",
        "🌤️ What time of year did you spot it?",
    ],
}

QUESTION_ORDER = [
    "cap-diameter", "cap-shape", "cap-surface", "cap-color",
    "does-bruise-or-bleed", "gill-attachment", "gill-color",
    "stem-height", "stem-width", "stem-color",
    "has-ring", "ring-type", "habitat", "season",
]

ACK_PHRASES = [
    "Got it! 👍", "Thanks! 🙌", "Noted! ✅", "Perfect! 😊",
    "Interesting! 🧐", "Okay, noted! 📝", "Good to know! 🌿",
    "Understood! 🔍", "Helpful! 💡", "Awesome! ⭐",
]


# ─────────────────────────────────────────────────────────────────────────────
# Model (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🍄 Training the mushroom model — just a moment…")
def load_model_and_encoders(csv_path="Mushroom_data.csv"):
    df = pd.read_csv(csv_path)
    drop_cols = ["stem-root", "veil-type", "veil-color", "spore-print-color", "stem-surface"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    for col in ["cap-surface", "gill-spacing"]:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    for col in ["gill-attachment", "ring-type"]:
        if col in df.columns:
            df[col] = df[col].bfill()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    le_target = LabelEncoder()
    df["class"] = le_target.fit_transform(df["class"])   # e=0, p=1

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    num_cols = [c for c in ["cap-diameter", "stem-height", "stem-width"] if c in df.columns]
    scaler = RobustScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = XGBClassifier(
        n_estimators=150, learning_rate=0.06, max_depth=2,
        subsample=0.7, colsample_bytree=0.6,
        use_label_encoder=False, eval_metric="logloss", random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf, encoders, scaler, le_target, X.columns.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "chat_history": [], "stage": "welcome",
        "q_index": 0, "answers": {},
        "used_variants": {}, "prediction_count": 0,
        "last_label": None, "last_confidence": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_msg(role, text):
    st.session_state.chat_history.append({"role": role, "text": text})


def get_question(feature):
    pool = QUESTION_POOL.get(feature, [f"What is the {feature}?"])
    used = st.session_state.used_variants.get(feature, set())
    available = [i for i in range(len(pool)) if i not in used]
    if not available:
        available = list(range(len(pool)))
        used = set()
    idx = random.choice(available)
    used.add(idx)
    st.session_state.used_variants[feature] = used
    return pool[idx]


def render_chat():
    for msg in st.session_state.chat_history:
        if msg["role"] == "bot":
            st.markdown(
                f'<div class="avatar">🍄</div>'
                f'<div class="bot-bubble">{msg["text"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="user-bubble">{msg["text"]}</div>'
                f'<div class="avatar" style="text-align:right">🧑</div>',
                unsafe_allow_html=True,
            )


def do_predict(answers, clf, encoders, scaler, feature_names):
    row = {}
    for feat in feature_names:
        val = answers.get(feat)
        if feat in FEATURE_META:
            le = encoders.get(feat)
            s = str(val) if val is not None else "f"
            try:
                row[feat] = le.transform([s])[0] if le else 0
            except ValueError:
                row[feat] = 0
        else:
            row[feat] = float(val) if val is not None else 0.0

    X = pd.DataFrame([row], columns=feature_names)
    num_cols = [c for c in ["cap-diameter", "stem-height", "stem-width"] if c in feature_names]
    if num_cols:
        X[num_cols] = scaler.transform(X[num_cols])

    prob = clf.predict_proba(X)[0]
    idx = int(np.argmax(prob))
    return ("edible" if idx == 0 else "poisonous"), float(prob[idx])


def advance(clf, encoders, scaler, feature_names):
    st.session_state.q_index += 1
    if st.session_state.q_index >= len(QUESTION_ORDER):
        label, conf = do_predict(
            st.session_state.answers, clf, encoders, scaler, feature_names)
        st.session_state.last_label = label
        st.session_state.last_confidence = conf
        st.session_state.prediction_count += 1
        st.session_state.stage = "result"
        if label == "edible":
            add_msg("bot",
                f"✅ Good news! Based on what you've told me, this mushroom "
                f"**looks edible** (confidence: {conf*100:.1f}%).\n\n"
                "⚠️ *Always double-check with a local expert before eating any wild mushroom!*")
        else:
            add_msg("bot",
                f"☠️ **Warning!** The features you described suggest this mushroom is "
                f"**poisonous** (confidence: {conf*100:.1f}%).\n\n"
                "🚫 *Please do NOT eat it. Stay safe out there!*")
    else:
        next_feat = QUESTION_ORDER[st.session_state.q_index]
        add_msg("bot", f"{random.choice(ACK_PHRASES)}  {get_question(next_feat)}")
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
init_state()

st.markdown("""
<div class="main-header">
  <h1 style="color:#a0c4ff;margin:0;font-size:2rem;">🍄 Mushroom Safety Bot</h1>
  <p style="color:#7fb3d3;margin:6px 0 0;font-size:.95rem;">
    Tell me about your mushroom and I'll predict whether it's safe to eat!
  </p>
</div>
""", unsafe_allow_html=True)

try:
    clf, encoders, scaler, le_target, feature_names = load_model_and_encoders()
except FileNotFoundError:
    st.error("❌ **Mushroom_data.csv not found.** "
             "Place it in the same directory as this script and restart.")
    st.stop()

# ── WELCOME ───────────────────────────────────────────────────────────────────
if st.session_state.stage == "welcome":
    if not st.session_state.chat_history:
        add_msg("bot",
            "👋 Hey there! I'm your friendly **Mushroom Safety Bot** 🍄\n\n"
            "I'll ask you a few simple questions about the mushroom you found, "
            "then tell you whether it's **safe to eat** or **potentially poisonous**.\n\n"
            "Ready to get started? Just hit the button below!")
    render_chat()
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Let's identify this mushroom!", use_container_width=True):
        st.session_state.stage = "asking"
        st.session_state.q_index = 0
        st.session_state.answers = {}
        add_msg("bot", get_question(QUESTION_ORDER[0]))
        st.rerun()

# ── ASKING ────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "asking":
    render_chat()
    current = QUESTION_ORDER[st.session_state.q_index]
    total = len(QUESTION_ORDER)
    st.markdown(
        f'<div class="step-indicator">📊 Question {st.session_state.q_index + 1} of {total}</div>',
        unsafe_allow_html=True)
    st.progress(st.session_state.q_index / total)

    if current in NUMERIC_FEATURES:
        units = {"cap-diameter": "cm", "stem-height": "cm", "stem-width": "mm"}
        defaults_n = {"cap-diameter": 5.0, "stem-height": 7.0, "stem-width": 10.0}
        unit = units[current]
        c1, c2 = st.columns([3, 1])
        with c1:
            num_val = st.number_input(
                f"Value ({unit})", min_value=0.1, max_value=500.0,
                value=defaults_n[current], step=0.1, label_visibility="collapsed")
        with c2:
            if st.button("➡️ Next", use_container_width=True, key="num_btn"):
                add_msg("user", f"{num_val} {unit}")
                st.session_state.answers[current] = num_val
                advance(clf, encoders, scaler, feature_names)
    else:
        opts = list(FEATURE_META[current]["options"].keys())
        n_cols = 3 if len(opts) > 4 else 2
        cols = st.columns(n_cols)
        for i, opt in enumerate(opts):
            with cols[i % n_cols]:
                if st.button(opt.capitalize(), key=f"o_{current}_{i}", use_container_width=True):
                    add_msg("user", opt.capitalize())
                    st.session_state.answers[current] = FEATURE_META[current]["options"][opt]
                    advance(clf, encoders, scaler, feature_names)

# ── RESULT ────────────────────────────────────────────────────────────────────
elif st.session_state.stage == "result":
    render_chat()
    label = st.session_state.last_label or "unknown"
    conf = st.session_state.last_confidence or 0.5
    bar = f"{conf*100:.0f}%"

    if label == "edible":
        st.markdown(
            f'<div class="edible-card">✅ EDIBLE<br>'
            f'<span style="font-size:14px;opacity:.9">Confidence: {conf*100:.1f}%</span>'
            f'<div class="confidence-bar"><div class="confidence-fill" style="width:{bar}"></div></div>'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="poisonous-card">☠️ POISONOUS<br>'
            f'<span style="font-size:14px;opacity:.9">Confidence: {conf*100:.1f}%</span>'
            f'<div class="confidence-bar"><div class="confidence-fill" style="width:{bar}"></div></div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Identify another mushroom", use_container_width=True):
            add_msg("bot", "🍄 Great — let's look at another one! "
                    "I'll mix up my questions a bit this time 😊")
            st.session_state.stage = "asking"
            st.session_state.q_index = 0
            st.session_state.answers = {}
            add_msg("bot", get_question(QUESTION_ORDER[0]))
            st.rerun()
    with c2:
        if st.button("🗑️ Start fresh", use_container_width=True):
            for k in ["chat_history","stage","q_index","answers",
                      "used_variants","prediction_count","last_label","last_confidence"]:
                st.session_state.pop(k, None)
            st.rerun()

    with st.expander("📋 View your answers"):
        for feat in QUESTION_ORDER:
            raw = st.session_state.answers.get(feat, "N/A")
            if feat in FEATURE_META:
                rev = {v: k for k, v in FEATURE_META[feat]["options"].items()}
                display = rev.get(str(raw), str(raw)).capitalize()
            else:
                display = str(raw)
            st.write(f"**{feat.replace('-',' ').title()}**: {display}")

    count = st.session_state.prediction_count
    if count > 1:
        st.markdown(
            f'<p style="text-align:center;color:#7fb3d3;font-size:13px;">'
            f"You've identified {count} mushrooms this session! 🍄</p>",
            unsafe_allow_html=True)

    st.markdown("""
<br>
<div style="text-align:center;color:#7fb3d3;font-size:12px;">
⚠️ This is a machine-learning prediction for <strong>educational purposes only</strong>.<br>
Never rely solely on this tool to decide whether a mushroom is safe to eat.<br>
Always consult a qualified mycologist.
</div>""", unsafe_allow_html=True)
