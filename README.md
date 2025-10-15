# A/B Test Analyzer

A local Streamlit app to compute statistical significance, confidence intervals, effect sizes, and recommend winners for A/B tests (binary and continuous metrics).

## Run locally
1. python -m venv venv
2. source venv/bin/activate   # or venv\Scripts\activate on Windows
3. pip install -r requirements.txt
4. streamlit run src/app.py

## CSV formats
- Binary: user_id,variant,converted  (converted: 0 or 1)
- Continuous: user_id,variant,revenue (floats)

