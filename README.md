# ML Pipeline: Iris Dataset

A concise and modular **Machine Learning pipeline** using the famous **Iris dataset** as an example. This project walks through:

1. **Data Ingestion**  
2. **Data Cleaning & Preprocessing**  
3. **Model Training & Evaluation**  
4. **Deployment** (via a Flask API)  
5. (Optional) **Exploratory Data Analysis** with Jupyter Notebook

```ascii
  +--------------+
  |   Dataset    |
  +--------------+
         v
  +--------------+
  | Data         |
  | Ingestion    |
  +--------------+
         v
  +---------------------+
  | Data Cleaning &     |
  | Preprocessing       |
  +---------------------+
         v
  +----------------------+
  | Model Training       |
  | & Evaluation         |
  +----------------------+
         v
  +--------------+
  | Deployment   |
  | (Flask API)  |
  +--------------+
         v
  [Model in Production]
```

## Contents

1. [Project Structure](#project-structure)  
2. [Requirements & Setup](#requirements--setup)  
3. [Usage](#usage)  
   - [Preprocessing](#1-preprocessing)  
   - [Training](#2-training)  
   - [Local Inference](#3-local-inference)  
   - [Running the Flask API](#4-running-the-flask-api)  
4. [Notebook EDA](#notebook-eda)  
5. [Project Explanation](#project-explanation)  
   - [Data Ingestion](#data-ingestion)  
   - [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)  
   - [Model Training & Evaluation](#model-training--evaluation)  
   - [Deployment (Flask)](#deployment-flask)  
6. [Future Improvements](#future-improvements)  
7. [License](#license)

## Project Structure

```
ml-pipeline/
├── app/
│   └── app.py              # Flask app for model deployment
├── data/
│   ├── raw/               # Original dataset (e.g., iris.csv)
│   └── processed/         # Processed data outputs
├── models/
│   └── best_model.pkl     # Serialized model + scaler + encoder
├── notebooks/
│   └── iris_eda.ipynb     # Jupyter Notebook with data exploration
├── src/
│   ├── ingestion.py       # (Optional) If you download data programmatically
│   ├── preprocessing.py   # Cleans data, applies scaling/encoding
│   ├── train.py          # Trains models, saves best one
│   └── predict.py        # Loads the best model, makes a sample prediction
├── venv/                  # Local virtual environment (not committed)
├── requirements.txt       # List of dependencies
└── README.md             # This file
```

## Requirements & Setup

**Recommended**: Use a **virtual environment** to isolate project dependencies.

1. **Clone this repository**:
   ```bash
   git clone https://github.com/biohazel/ml-pipeline.git
   cd ml-pipeline
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or .\venv\Scripts\activate on Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Register your venv as a Jupyter kernel**:
   ```bash
   python -m ipykernel install --user --name ml-pipeline-venv --display-name "Python (ml-pipeline)"
   ```

## Usage

### 1. Preprocessing

```bash
python3 src/preprocessing.py
```
- Loads data/raw/iris.csv
- Encodes categorical labels into numeric form
- Scales numeric features
- Splits data into train/test
- (Optionally) saves processed splits to data/processed/

### 2. Training

```bash
python3 src/train.py
```
- Trains multiple models (e.g., Logistic Regression and RandomForestClassifier)
- Compares them on test accuracy
- Saves the best model, along with the scaler and label encoder, to models/best_model.pkl

### 3. Local Inference

```bash
python3 src/predict.py
```
- Loads models/best_model.pkl
- Predicts on a hardcoded sample [5.1, 3.5, 1.4, 0.2]
- Prints the predicted species to the console

### 4. Running the Flask API

```bash
python3 app/app.py
```
- Launches a local Flask server on http://127.0.0.1:5000/
- Only accepts POST requests at /predict

Sample test with curl:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"features":[5.1, 3.5, 1.4, 0.2]}' \
     http://127.0.0.1:5000/predict
```
Expects a JSON response like `{"species":"Iris-setosa"}`

## Notebook EDA

In the `notebooks/iris_eda.ipynb`, you'll find:

- Initial Data Exploration: `df.head()`, `df.describe()`, `df.info()`
- Missing Value Checks: `df.isnull().sum()`
- Basic Visuals: Histograms, correlation heatmap, etc.
- Conclusions about data distributions and potential modeling implications

To run the notebook:
```bash
jupyter notebook
```
Then open `iris_eda.ipynb` in your browser.
(If you registered your kernel, select `Python (ml-pipeline)` so it uses the correct environment.)

## Project Explanation

### Data Ingestion
- Purpose: Acquire or read the Iris dataset (CSV)
- Often done by `preprocessing.py` or a separate `ingestion.py` script if you're downloading data from a URL

### Data Cleaning & Preprocessing
- Purpose: Remove noise, handle missing values (none in Iris), encode labels, scale numeric features
- Ensures consistent data for the ML algorithms

### Model Training & Evaluation
`train.py`:
- Runs Logistic Regression and Random Forest
- Measures accuracy via `accuracy_score`
- Picks the best model, saves it as `best_model.pkl`

### Deployment (Flask)
`app.py`:
- Loads the saved model, scaler, and label encoder
- Defines a `/predict` endpoint (POST) for real-time inference
- Returns predicted species in JSON format

## Future Improvements

- **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to fine-tune model parameters
- **Dockerization**: Add a Dockerfile to containerize the entire application
- **CI/CD**: Configure GitHub Actions or similar for automated tests, linting, or deployments
- **Logging & Monitoring**: Track prediction requests, response times, and error rates in production


