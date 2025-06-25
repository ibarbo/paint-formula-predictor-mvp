# Paint Formula Success Predictor MVP

## 🎯 Project Overview

This project focuses on developing a Machine Learning model to predict the success or failure of paint formulas based on their chemical composition, properties, and application conditions. The primary goal is to accelerate the R&D process, optimize resource allocation, and reduce the time-to-market for new paint products by intelligently identifying promising formulas.

This repository showcases a comprehensive ML project lifecycle, emphasizing data generation, robust feature engineering, advanced modeling techniques, and meticulous evaluation, especially in the face of challenging imbalanced datasets.

## 💡 The Business Problem

In the paint manufacturing industry, developing new formulas is a complex, time-consuming, and resource-intensive process. Many experimental formulas end up failing due to various factors (e.g., incorrect ratios, unsuitable additives, poor performance under specific conditions). Identifying successful formulas early can significantly reduce material waste, testing costs, and accelerate innovation.

Our aim is to build a predictive tool that can guide R&D chemists towards more promising formulations.

## 📊 Data Strategy & Evolution

Given the initial absence of real-world datasets, a **simulated data approach** was implemented.

1.  **Initial Data Simulation:** A basic dataset was created to establish the project pipeline, simulating fundamental paint components and properties.
2.  **Advanced Data Simulation:** Recognizing the limitations of the initial simplistic data (which led to underperforming models), a **more complex and realistic data simulator** (`scripts/data_generation/simulate_complex_paint_formulas.py`) was developed. This simulator incorporates a wider range of components (e.g., multiple additives), application conditions (temperature, drying time), substrate types, and intricate logical relationships governing the `IsSuccess` outcome. This iterative improvement of the data source was crucial for enhancing the model's potential.

## 🛠️ Technical Stack

* **Languages:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imblearn`, `matplotlib`, `seaborn`

## ✨ Key Features & Methodologies

* **End-to-End ML Pipeline:** Structured approach covering data generation, EDA (conceptual), preprocessing, feature engineering, modeling, hyperparameter tuning, and robust evaluation.
* **Comprehensive Data Preprocessing:**
    * Handling of categorical features using One-Hot Encoding.
    * Numerical feature scaling (StandardScaler).
    * Data splitting for robust model evaluation.
* **Advanced Feature Engineering:** Creation of domain-inspired features crucial for capturing complex relationships:
    * **Component Ratios:** `ResinToPigmentRatio`, `ResinToSolventRatio` (reflecting critical formulation balances).
    * **Total Additive Content:** `TotalAdditivesPercentage` (capturing overall additive strategy).
    * **Substrate-Component Interactions:** `AcrylicOnWood`, `EpoxyOnMetal`, `TiO2OnConcrete` (modeling performance synergy/conflict with specific application surfaces).
    * **Estimated Density:** `EstimatedDensity` (a proxy for a key physical property affecting application).
    * **Extreme Condition Indicators:** `HighDryingTime`, `LowApplicationTemp` (flagging challenging application scenarios).
* **Robust Classification Models:**
    * **XGBoost:** Utilized for its high performance, regularization capabilities, and ability to handle complex non-linear relationships.
    * **Random Forest:** Employed as a strong baseline and alternative ensemble method.
* **Addressing Class Imbalance:** A critical challenge in this project (`IsSuccess` is a minority class).
    * **SMOTE (Synthetic Minority Over-sampling Technique):** Applied to balance the training dataset.
    * **`scale_pos_weight` (XGBoost):** Configured to penalize errors on the minority class more heavily during training.
    * **Threshold Optimization:** A custom threshold adjustment strategy was implemented to maximize the F1-score for the minority class, prioritizing the identification of `Success` formulas.
* **Rigorous Model Evaluation:** Beyond simple accuracy, focus on:
    * **Precision, Recall, F1-score:** Crucial for imbalanced datasets, especially for the `Success` class.
    * **AUC-ROC:** Assessing the model's overall discriminative power across various thresholds.
    * **Detailed Confusion Matrix Analysis:** Deep dive into False Positives (wasted resources) and False Negatives (missed opportunities).
* **Iterative Improvement:** The project demonstrates an iterative cycle of model evaluation, diagnosis (identifying high bias/underfitting), and strategic data/feature enhancement.

## 📈 Current Status & Results

The model is currently undergoing re-training with the newly generated, more complex data and engineered features.

* **Initial Findings (before new data/FE):** Models exhibited significant underfitting (high bias), with AUC-ROC scores consistently below 0.5 (worse than random chance for discrimination). Even aggressive threshold adjustment resulted in impractically high False Positives. This highlighted the critical need for more informative data and features.
* **Expected Outcome (with new data/FE):** We anticipate a substantial improvement in model performance, particularly in its ability to correctly identify `Success` formulas (higher Recall and a more balanced Precision/Recall trade-off), and an AUC-ROC significantly above 0.5, indicating true discriminative power.

## 🚀 Future Work & Next Steps

Upon successful re-training and evaluation, the roadmap for this project includes:

1.  **Deployment Strategy:** Planning how this predictive model could be integrated into the R&D workflow (e.g., web application, internal tool).
2.  **Real-World Data Integration:** Transitioning from simulated data to actual historical paint formula data, which is paramount for a production-ready solution. This includes planning data collection, cleaning, and integration pipelines.
3.  **Advanced Model Interpretability:** Implementing techniques like SHAP or LIME to provide chemists with insights into *why* a formula is predicted as successful, fostering trust and aiding in knowledge discovery.
4.  **Continuous Monitoring & Retraining:** Establishing a system to monitor model performance in production and a pipeline for periodic retraining with new data to prevent model drift.
5.  **Collaboration with Domain Experts:** Close collaboration with R&D chemists to refine feature engineering, validate model insights, and define clearer success metrics.

## 🤝 How to Run the Project

1.  Clone this repository: `git clone [repository-url]`
2.  Navigate to the project directory: `cd paint_formula_success_predictor_mvp`
3.  Create and activate a virtual environment (recommended).
4.  Install dependencies: `pip install -r requirements.txt`
5.  Run the scripts in sequence from the project root:
    ```bash
    python scripts/data_generation/simulate_complex_paint_formulas.py
    python scripts/preprocessing/feature_engineering.py
    python scripts/preprocessing/data_preprocessing.py
    python scripts/modeling/paint_formula_xgboost_modeling.py
    ```
    (Note: The last script may take significant time due to GridSearchCV).

---