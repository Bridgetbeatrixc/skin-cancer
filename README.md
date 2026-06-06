# Skin Lesion Classifier Streamlit App

Educational Streamlit demo for a HAM10000 7-class skin lesion classifier trained with TensorFlow/Keras and EfficientNetB0.

Disclaimer: This model is for educational purposes only and is not a medical diagnosis tool.

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The model file should be here:

```text
models/skin_cancer_efficientnetb0.keras
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Make sure the repository includes:
   - `app.py`
   - `requirements.txt`
   - `models/skin_cancer_efficientnetb0.keras`
3. Go to Streamlit Community Cloud.
4. Create a new app from the GitHub repository.
5. Open **Advanced settings** and set Python to **3.11**. TensorFlow will not install on Python 3.14.
6. Set the main file path to:

```text
app.py
```

Note: the model is around 37 MB, which is usually acceptable for GitHub and Streamlit Cloud. If deployment becomes slow, use Git LFS or download the model from a release asset.

If Streamlit Cloud already created the app with Python 3.14, delete the app and redeploy it. Rebooting alone will keep the same Python environment.

## Optional Gemini Explanation

The app can generate a context-aware educational explanation with Gemini. Do not put the API key in the code.

For local PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_PRIVATE_KEY"
streamlit run app.py
```

For Streamlit Community Cloud, add this in the app secrets:

```toml
GEMINI_API_KEY = "YOUR_PRIVATE_KEY"
```

If no Gemini key is configured, the app uses a local fallback explanation.
