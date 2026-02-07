# App User Segmentation using Unsupervised Machine Learning

## 📌 Project Overview
This project analyzes mobile application user behavior using unsupervised machine learning techniques.  
The goal is to segment users based on engagement patterns and identify high-value and at-risk users.

K-Means clustering is applied after data preprocessing and feature scaling.  
The project produces meaningful customer segments for business decision-making.

---

## 🎯 Objectives
- Segment users based on usage behavior
- Identify high, moderate, and low engagement users
- Detect churn-risk customers
- Support personalized marketing strategies

---

## 📊 Dataset
The dataset contains:
- User demographics
- Device information
- Session activity
- Engagement metrics
- Churn indicators

Location:
data/app_user_behavior_dataset.csv


---

## 🛠️ Tech Stack
- Python
- Pandas
- Scikit-learn
- Matplotlib
- NumPy

---

## 📂 Project Structure
APP_USER_SEGMENTATION/
│
├── data/
├── outputs/
│ └── plots/
├── src/
│ ├── data_processing.py
│ ├── feature_engineering.py
│ ├── clustering.py
│ └── evaluation.py
├── main.py
└── README.md


---

## 🔍 Workflow

1. Data Cleaning
   - Handle missing values
   - Remove duplicates

2. Feature Engineering
   - Select numerical features
   - Apply StandardScaler

3. Model Selection
   - Elbow Method to choose optimal K

4. Clustering
   - Apply K-Means

5. Evaluation
   - PCA visualization
   - Cluster profiling

6. Output Generation
   - Save clustered data
   - Save plots

---

## 🚀 How to Run

Install dependencies:

pip install -r requirements.txt


Run project:

python main.py


---

## 📈 Outputs

- Clustered Dataset:
outputs/clustered_users.csv


- Plots:
outputs/plots/elbow_method.png
outputs/plots/pca_clusters.png


---

## 📌 Results

Users were segmented into four clusters:

- High Engagement Users
- Moderate Users
- Low Engagement / At-Risk Users
- Occasional Users

The model successfully identified behavioral patterns for business action.

---

## 👤 Author
Vijay Ragavan
Aspiring Data Scientist
