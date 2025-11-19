# malicious_url_detection
Real-Time Malicious URL Detection in a Chromium-Compatible Browser Extension Using Heuristics, NLP, Random Forest, and Explainable AI (XAI)


Folder  Structure of this project 


Malicious_URL_Detection_Project/
│
├── 0_data/
│   ├── raw/                 # Original datasets (unaltered)
│   │   ├── alexa.csv
│   │   ├── phishtank.csv
│   │   └── kaggle.csv
│   └── processed/           # Cleaned, merged, labeled datasets
│       └── cleaned_urls.csv
│
├── 1_notebooks/             # For research, visualization, and experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_experiment.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_model_experiment.ipynb
│
├── 2_scripts/               # Production-ready Python scripts
│   ├── preprocessing.py      # Clean, label, merge datasets
│   ├── feature_extraction.py # Heuristic + NLP features
│   ├── train_model.py        # Train Random Forest / alternative ML models
│   ├── evaluate_model.py     # Compute metrics: accuracy, precision, recall, F1
│   └── xai_analysis.py       # LIME explainability and plots
│
├── 3_models/                # Saved ML models & vectorizers
│   ├── rf_model.pkl
│   ├── xgb_model.pkl        # Optional: XGBoost
│   └── vectorizer.pkl       # Saved vectorizer for NLP features
│
├── 4_features/              # Extracted features
│   ├── heuristic_features.csv
│   ├── nlp_features.csv
│   └── combined_features.csv # Final dataset for model training
│
├── 5_results/               # All outputs, plots, metrics, and reports
│   ├── metrics/
│   │   └── rf_metrics.json
│   ├── plots/
│   │   ├── lime_example1.png
│   │   └── lime_example2.png
│   └── reports/
│       └── final_report.pdf
│
├── 6_extension/             # Chromium browser extension
│   ├── manifest.json
│   ├── background.js
│   ├── content.js
│   ├── popup.html
│   ├── popup.js
│   └── model_integration.py  # Load trained model & predict URLs
│
├── 7_api/                   # FastAPI backend for real-time predictions
│   ├── app.py               # Main FastAPI app
│   ├── routes.py            # API endpoints: /predict, /explain
│   ├── model_loader.py      # Load ML model & vectorizer
│   └── requirements_api.txt # FastAPI-specific dependencies
│
├── requirements.txt         # Full Python dependencies for the project
├── README.md                # Project overview & instructions
└── main.py                  # Optional orchestrator for entire pipeline
