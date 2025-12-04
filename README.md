# 🍌🍎 Fruit Freshness Classification using Machine Learning

This project focuses on building a **Machine Learning model** capable of classifying fruit images into:

- **Fresh Apples**
- **Fresh Bananas**
- **Fresh Oranges**
- **Rotten Apples**
- **Rotten Bananas**
- **Rotten Oranges**

The project was built from scratch with manually extracted features, aiming to test how powerful ML models can be when the data is well-engineered — even without using Deep Learning.

---

## 📌 Project Goals

The main purpose of this project was **educational**, aiming to:

- Practice building an end-to-end ML system from scratch  
- Learn **Feature Engineering** in depth  
- Apply **Cross Validation** (Stratified K-Fold)  
- Use ML **Pipelines** correctly  
- Test whether ML models can achieve high accuracy as an alternative to DL in some cases  
- Prepare and clean the dataset manually  

---

## 📂 Dataset

The project uses the following public dataset from Kaggle:

🔗 **Fruits Fresh and Rotten for Classification**  
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification/data

The dataset contains real images of apples, bananas, and oranges — both fresh and rotten — divided into `train` and `test` folders.

---

## 🧪 Feature Engineering

Instead of feeding raw images into a neural network, this project extracts **16 handcrafted features** from each image:

### **Color Features (RGB)**
- mean_R, mean_G, mean_B  
- std_R, std_G, std_B  

### **Color Features (HSV)**
- mean_H, mean_S, mean_V  
- std_H, std_S, std_V  

### **Image Structure**
- dark_ratio  
- high_saturation_ratio  
- gray_std  
- laplacian_var (image sharpness)  

These features are generated using **OpenCV** and saved into CSV files:

- `fruits_train_features.csv`
- `fruits_test_features.csv`

---

## ⚙️ Model Training

Three ML models were trained and evaluated:

- **KNN**
- **SVM (RBF kernel)**
- **Random Forest**

Each model was built inside a **Pipeline**:

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('model', ...)
])
```

To ensure reliable evaluation, **Stratified K-Fold Cross Validation** was applied (5 folds).

---

## 🏆 Results

| Model          | Mean CV Accuracy |
|----------------|------------------|
| KNN            | ~88%             |
| SVM (RBF)      | ~87%             |
| Random Forest  | **~96%**         |

Final test accuracy for the best model:

### 🎯 **Random Forest Test Accuracy: 97.1%**

This demonstrates that with strong Feature Engineering, ML models can achieve excellent results — sometimes comparable to DL models — for specific types of tasks.

---

## 🧩 Predicting New Images

To classify new custom images:

1. Place images inside a folder named:  
   `images/`

2. Run the feature extractor script to generate:  
   `new_images_features.csv`

3. Load the file and use the trained model to predict labels.

Example:

```python
new_data = pd.read_csv("new_images_features.csv")
X_new = new_data[feature_names]
predictions = best_model.predict(X_new)
```

---

## 📁 Folder Structure

```
📦 Fruit-Classification-ML-Project
├── ExtractingFeatures.ipynb
├── fruits_train_features.csv
├── fruits_test_features.csv
├── new_images_features.csv
├── model_training.ipynb
├── README.md
└── images/
```

---

## 🛠️ Technologies Used

- Python  
- NumPy  
- Pandas  
- OpenCV  
- Scikit-Learn  
- Kaggle Dataset  

---

## 📌 Key Learnings

- Importance of **feature engineering**  
- Using **cross validation** correctly  
- How ML pipelines improve consistency  
- When ML can replace DL in practical scenarios  
- Impact of data quality on model accuracy  

---

## 🔗 Repository

👉 **GitHub Link:**  
https://github.com/enghasan1/Fruits-fresh-and-rotten-for-classification

---

## 👨‍💻 Author

**Hasan Receb**  

