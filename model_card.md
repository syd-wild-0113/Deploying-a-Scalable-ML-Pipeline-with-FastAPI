# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* **Developer:** Sydney Wildeboer
* **Model Date:** July 2026
* **Model Version:** 1.0.0
* **Model Type:** Random Forest Classifier
* **Hyperparameters:** n_estimators=100, max_depth=15, random_state=42
* **Artifacts:** Serialized model (model.pkl), OneHotEncoder (encoder.pkl), and LabelBinarizer(lb.pkl)

## Intended Use
* **Primary Intended Use:** Predict whether an individual's annual income exceeds $50,000 based on demographic, educational, and employment attributes.
* **Intended Users:** Data science students, automated reporting systems, and API consumer applications evaluating scalable model deployments.
* **Out-of-Scope Uses:** This model should not be used for real-world employment decisions, credit lending evaluations, or individual financial assessments, as it was trained strictly on historical 1994 US Census data.

## Training Data
* **Source:** 1994 US Census Bureau Income Dataset (commonly known as the "Adult" dataset).
* **Train / Test Split:** 80% of the dataset was reserved for training (~26,048 samples).
* **Pre-Processing:** 
  * Cleaned leading/trailing whitespace from string entries.
  * Continuous features were used directly.
  * Categorical features (workclass, education, marital-status, occupation, relationship, race, sex, native-country) were processed using OneHotEncoder(sparse=False, handle_unknown="ignore").
  * Target variable (salary) was binarized using LabelBinarizer where >50K maps to 1 and <=50K maps to 0.
  
## Evaluation Data
* **Test Set:** 20% of the dataset was reserved for model evaluation (~6,513 samples).
* **Pre-Processing:** Evaluated using the exact encoder and label binarizer fitted on the training split (training=False).

## Metrics
* **Overall Model Performance:**
  * **Precision:** *0.7918*
  * **Recall:** *0.5786*
  * **F1-Score:** *0.6686*

* **Key Takeaway:** The model achieved **strong overall precision**, but **underperforms on minority income slices**.

## Ethical Considerations
* **Fairness and Bias:** The dataset originates from 1994 census data, reflecting historical socioeconomic disparities across gender, race, and nationality. Subgroup evaluations show variance in recall across different demographic slices (e.g., gender and race categories).
* **Privacy:** All records are anonymized public census records without personally identifiable information (PII).

## Caveats and Recommendations
* **Outdated Data:** The data represents economic and demographic distributions from 1994 and does not reflect current wage levels, inflation, or job classifications.
* **Class Imbalance:** The target variable is imbalanced, with roughly 75% of entries making <=50K and 25% making >50K. Further iterations could benefit from class weighting (class_weight="balanced") or threshold tuning to boost recall for high earners.