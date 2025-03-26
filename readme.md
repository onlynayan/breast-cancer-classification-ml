# 🩺 Breast Cancer Classification Project

This project uses supervised machine learning models to classify breast tumors as **benign** or **malignant** using the **Breast Cancer Wisconsin (Diagnostic) Dataset**. The workflow includes data preprocessing, model training with multiple classifiers, evaluation using metrics and confusion matrices, and meaningful visualizations.

---

## 📚 About the Dataset

The dataset consists of features computed from digitized images of **fine needle aspirates (FNA)** of breast masses. These features describe the characteristics of the **cell nuclei** present in the image. No missing values are present in the dataset.

📁 **Source**:  
- UCI Machine Learning Repository: [Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

📊 **Class Distribution**:
- **Benign**: 357 samples  
- **Malignant**: 212 samples  

🔢 **Features**:  
30 real-valued features derived from:
- **Radius**
- **Texture**
- **Perimeter**
- **Area**
- **Smoothness**
- **Compactness**
- **Concavity**
- **Concave points**
- **Symmetry**
- **Fractal dimension**  

Each of the above is computed as:
- Mean
- Standard Error
- Worst (largest mean of top 3 values)

---

## 🛠️ Project Workflow

### ✅ 1. Data Preprocessing
- Loaded and explored the dataset
- Dropped irrelevant columns like `ID`
- Converted diagnosis labels to binary format (`M` = 1, `B` = 0)

### ✅ 2. Feature Scaling
- Applied `StandardScaler` to normalize feature values before model training

### ✅ 3. Train-Test Split
- Split the data into **training** and **testing** sets (80/20)

### ✅ 4. Predictive Modeling
Trained and evaluated the following classification models:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting Classifier**

### ✅ 5. Evaluation & Validation
- Calculated **accuracy** for all models
- Plotted **confusion matrices** to visualize classification performance
- Used **cross-validation** to ensure generalizability
- Generated a **classification report** with precision, recall, and F1-score

---

## 📈 Visualizations

- 📊 **Confusion Matrix Heatmaps** – to understand TP, TN, FP, FN for each model  
- 🎯 **Scatter Plot** – compared predicted vs actual labels  
- ❌ **Misclassification Plot** – highlighted wrongly predicted instances  

---

## ✅ Conclusion

This project demonstrates how multiple machine learning algorithms can be trained and compared for a binary classification problem in healthcare. Logistic Regression, SVM, and Gradient Boosting performed exceptionally well, with accuracy above **95%**, and minimal misclassifications.

---

## 💡 Future Improvements

- Implement ROC-AUC curve for more nuanced model comparison
- Explore SHAP or LIME for model interpretability
- Deploy model using Streamlit or Flask for real-time predictions

---

## 📌 Tools & Libraries Used

- Python, Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (LogisticRegression, SVC, KNN, RandomForest, etc.)
- UCI ML Repository Dataset

---

## 🙋‍♂️ Author

**onlynayan**  
Aspiring Data Scientist | Passionate about AI in Healthcare  

