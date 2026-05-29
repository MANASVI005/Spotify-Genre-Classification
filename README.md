 🎵 Spotify Genre Classification

A machine learning project that predicts the **music genre** of Spotify tracks based on audio features and text metadata using **LightGBM** + **TF-IDF** + **Optuna** hyperparameter tuning.


📌 Project Overview

This project tackles multi-class genre classification on Spotify track data. It combines structured audio features with text-based TF-IDF features from track metadata, tuned using Optuna, and evaluated with Stratified K-Fold cross-validation.


 📁 Repository Structure

Spotify-Genre-Classification/
│
├── spotify_genre.ipynb        # Main ML notebook
├── confusion_matrix.png       # Model evaluation visualization
├── report.txt                 # Results and findings
├── submission.csv             # Final predictions
├── optuna_study.pkl           # Hyperparameter tuning study
│── scaler.pkl             # Feature scaler
│── label_encoder.pkl      # Label encoder
│── tfidf_vectorizer.pkl   # TF-IDF vectorizer
│
├── backend/                   # 🚧 Coming Soon — Flask/FastAPI API
└── frontend/                  # 🚧 Coming Soon — Web UI
```

---

🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| LightGBM | Primary classification model |
| Optuna | Hyperparameter optimization |
| Scikit-learn | Preprocessing, cross-validation |
| TF-IDF | Text feature extraction |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |

 ⚙️ How to Run

1. Clone the repository
git clone https://github.com/MANASVI005/Spotify-Genre-Classification.git
cd Spotify-Genre-Classification

2. Install dependencies
pip install -r requirements.txt

3. Add the data files
Download `train.csv` and `test.csv` and place them in the root folder.
> These are not included in the repo due to large file size.

 4. Run the notebook
Open `spotify_genre.ipynb` in Jupyter Notebook or VS Code and run all cells.



 📊 Model Pipeline

Raw Data
   ↓
Data Cleaning & Preprocessing
   ↓
Feature Engineering (Audio + TF-IDF Text Features)
   ↓
Optuna Hyperparameter Tuning
   ↓
LightGBM with Stratified K-Fold CV
   ↓
Final Predictions → submission.csv


📈 Results

- **Model:** LightGBM (LGBM)
- **Validation Strategy:** Stratified K-Fold
- **Tuning:** Optuna with automated hyperparameter search
- See `report.txt` for detailed metrics and findings
- See `confusion_matrix.png` for class-wise performance

---

 🚫 Files Not Included

Due to GitHub file size limits, the following are excluded:

| File | Reason |
|------|--------|
| `trained_model.pkl` (1.1 GB) | Too large — retrain using notebook |
| `train.csv` (14 MB) | Original dataset |
| `test.csv` (5 MB) | Original dataset |
| `X_final.npz` (12 MB) | Intermediate feature matrix |
| `*_test_preds.pkl` (28 MB) | Intermediate predictions |

---

🔮 Roadmap

- [x] Data preprocessing & feature engineering
- [x] LightGBM model with Optuna tuning
- [x] Cross-validation & evaluation
- [ ] Backend API (Flask / FastAPI)
- [ ] Frontend Web UI
- [ ] Model deployment

---

## 👩‍💻 Author

**Manasvi**  
GitHub: [@MANASVI005](https://github.com/MANASVI005)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
