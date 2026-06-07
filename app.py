from pathlib import Path
import json
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="SK",
    layout="wide",
)


MODEL_PATH = Path("models/skin_cancer_efficientnetb0.keras")
IMG_SIZE = 224
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Place skin_cancer_efficientnetb0.keras in the models folder."
        )

    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except TypeError as exc:
        if "quantization_config" not in str(exc):
            raise

        patched_path = patch_keras_archive(MODEL_PATH)
        return tf.keras.models.load_model(patched_path, compile=False)


def remove_quantization_config(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        for child in value.values():
            remove_quantization_config(child)
    elif isinstance(value, list):
        for child in value:
            remove_quantization_config(child)


def patch_keras_archive(model_path: Path) -> str:
    """
    Some Kaggle/Keras versions save Dense layers with quantization_config=None.
    Older Keras 3 loaders may reject that key, so this creates a temporary
    compatible copy of the .keras archive.
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="skin_model_"))
    patched_path = temp_dir / "skin_cancer_efficientnetb0_compatible.keras"

    with zipfile.ZipFile(model_path, "r") as source_zip:
        with zipfile.ZipFile(patched_path, "w", compression=zipfile.ZIP_DEFLATED) as target_zip:
            for item in source_zip.infolist():
                content = source_zip.read(item.filename)

                if item.filename == "config.json":
                    config = json.loads(content.decode("utf-8"))
                    remove_quantization_config(config)
                    content = json.dumps(config).encode("utf-8")

                target_zip.writestr(item, content)

    return str(patched_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image).astype("float32")
    return np.expand_dims(image_array, axis=0)


def predict(image: Image.Image, model: tf.keras.Model) -> pd.DataFrame:
    image_batch = preprocess_image(image)
    probabilities = model.predict(image_batch, verbose=0)[0]

    results = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Description": [CLASS_DESCRIPTIONS[name] for name in CLASS_NAMES],
            "Confidence": probabilities,
        }
    )
    return results.sort_values("Confidence", ascending=False).reset_index(drop=True)


def get_gemini_api_key() -> str:
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""

    return os.environ.get("GEMINI_API_KEY", secret_key)


def get_configured_gemini_model_name() -> str:
    try:
        secret_model = st.secrets.get("GEMINI_MODEL_NAME", "")
    except Exception:
        secret_model = ""

    return os.environ.get("GEMINI_MODEL_NAME", secret_model)


def normalize_gemini_model_name(name: str) -> str:
    return name.removeprefix("models/")


def model_supports_generate_content(model) -> bool:
    methods = getattr(model, "supported_generation_methods", []) or []
    return "generateContent" in methods


def select_gemini_model(genai) -> str:
    configured_model = get_configured_gemini_model_name()
    if configured_model:
        return normalize_gemini_model_name(configured_model)

    available = [
        normalize_gemini_model_name(model.name)
        for model in genai.list_models()
        if model_supports_generate_content(model)
    ]

    for candidate in GEMINI_MODEL_CANDIDATES:
        if candidate in available:
            return candidate

    for model_name in available:
        if "flash" in model_name:
            return model_name

    if available:
        return available[0]

    raise RuntimeError("No Gemini models supporting generateContent were found for this API key.")


def build_context(age, sex, outdoor_exposure, sunscreen_use, occupation, daily_life_notes):
    return {
        "age": age,
        "sex": sex,
        "outdoor_exposure": outdoor_exposure,
        "sunscreen_use": sunscreen_use,
        "occupation_or_daily_activity": occupation,
        "additional_daily_life_notes": daily_life_notes.strip() or "Not provided",
    }


def make_fallback_explanation(results: pd.DataFrame, context: dict) -> str:
    top = results.iloc[0]
    confidence = top["Confidence"] * 100

    return (
        f"The image model's top prediction is {top['Class']} "
        f"({top['Description']}) with {confidence:.2f}% confidence. "
        "The personal context can help frame risk factors, especially sun exposure, "
        "age, and sunscreen habits, but it does not confirm or rule out any condition. "
        f"In this input, the reported outdoor exposure is {context['outdoor_exposure']} "
        f"and sunscreen use is {context['sunscreen_use']}. "
        "If the lesion is new, changing, bleeding, painful, asymmetric, or concerning, "
        "the safest next step is professional evaluation by a dermatologist."
    )


def build_gemini_prompt(results: pd.DataFrame, context: dict) -> str:
    top_rows = results.head(3)
    prediction_lines = "\n".join(
        f"- {row['Class']} ({row['Description']}): {row['Confidence'] * 100:.2f}%"
        for _, row in top_rows.iterrows()
    )

    return f"""
You are helping explain an educational skin lesion image classification demo.
Do not diagnose. Do not claim the person has or does not have cancer.
Do not provide treatment instructions. Encourage professional medical evaluation for concerning lesions.

Model output:
{prediction_lines}

User context:
- Age: {context['age']}
- Sex: {context['sex']}
- Outdoor exposure: {context['outdoor_exposure']}
- Sunscreen use: {context['sunscreen_use']}
- Occupation or daily activity: {context['occupation_or_daily_activity']}
- Additional notes: {context['additional_daily_life_notes']}

Write a concise explanation in plain English with these sections:
1. What the model saw
2. How the context may matter
3. What to do next safely

Keep it under 180 words.
""".strip()


def generate_gemini_explanation(results: pd.DataFrame, context: dict) -> str:
    api_key = get_gemini_api_key()

    if not api_key:
        return make_fallback_explanation(results, context)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_name = select_gemini_model(genai)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(build_gemini_prompt(results, context))
        return response.text
    except Exception as exc:
        return (
            make_fallback_explanation(results, context)
            + f"\n\nGemini explanation was unavailable: {exc}"
        )


def main():
    st.title("Skin Lesion Image Classifier")
    st.caption("EfficientNetB0 model trained on HAM10000 for 7-class lesion image classification.")

    st.warning(
        "This model is for educational purposes only and is not a medical diagnosis tool. "
        "Do not use it to make health decisions."
    )

    with st.sidebar:
        st.header("Project")
        st.write("Upload a skin lesion image to run a prediction.")
        st.write("Classes: akiec, bcc, bkl, df, mel, nv, vasc.")
        st.divider()
        st.header("User Context")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        sex = st.selectbox("Sex", ["Prefer not to say", "Female", "Male", "Other"])
        outdoor_exposure = st.selectbox(
            "Daily outdoor exposure",
            ["Low", "Moderate", "High"],
            index=1,
        )
        sunscreen_use = st.selectbox(
            "Sunscreen use",
            ["Often", "Sometimes", "Rarely", "Not sure"],
        )
        occupation = st.text_input(
            "Daily activity or occupation",
            placeholder="Example: student, office worker, driver, outdoor worker",
        )
        daily_life_notes = st.text_area(
            "Additional daily life notes",
            placeholder="Example: frequent beach activity, sports, sunburn history",
            height=90,
        )
        st.caption("Context is used for educational explanation only. It does not make a diagnosis.")
        st.divider()
        st.write("Model path:")
        st.code(str(MODEL_PATH))

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
    )

    try:
        model = load_model()
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()

    if uploaded_file is None:
        st.info("Upload a JPG or PNG image to start.")
        st.stop()

    image = Image.open(uploaded_file)
    results = predict(image, model)
    top_result = results.iloc[0]
    user_context = build_context(
        age,
        sex,
        outdoor_exposure,
        sunscreen_use,
        occupation or "Not provided",
        daily_life_notes,
    )

    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)

    with right:
        st.subheader("Prediction")
        st.metric(
            label=f"Top class: {top_result['Class']}",
            value=f"{top_result['Confidence'] * 100:.2f}%",
        )
        st.write(top_result["Description"])

        chart_data = results.set_index("Class")["Confidence"]
        st.bar_chart(chart_data)

        display_results = results.copy()
        display_results["Confidence"] = display_results["Confidence"].map(lambda x: f"{x * 100:.2f}%")
        st.dataframe(display_results, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Interpretation Notes")
    st.write(
        "The confidence values show how the model distributes probability across the seven HAM10000 "
        "classes. A high confidence score does not mean the prediction is clinically correct."
    )

    st.divider()
    st.subheader("Context-Aware Educational Explanation")
    st.write(
        "This section combines the image model output with the user context to explain possible "
        "risk factors and next steps. It is not a diagnosis."
    )

    if st.button("Generate explanation with Gemini"):
        with st.spinner("Generating explanation..."):
            explanation = generate_gemini_explanation(results, user_context)
        st.write(explanation)


if __name__ == "__main__":
    main()
