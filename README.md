# NSL-KDD-Intrusion-Detection-System-Analysis
Comprehensive Report: NSL-KDD Intrusion Detection System Analysis
This report details the methodology, model evaluation, and interpretability insights derived from developing a Logistic Regression-based Intrusion Detection System (IDS) using the NSL-KDD dataset.

1. Data Loading and Feature Naming
The initial phase of the analysis involved loading the intrusion detection system (IDS) dataset. The data was sourced from the following public GitHub repository: https://github.com/Jehuty4949/NSL_KDD/blob/master/KDDTest%2B.csv. Given that the dataset was provided without explicit headers, a critical preliminary step was to assign meaningful and descriptive column names. This ensures clarity in subsequent data manipulation, feature selection, and model interpretation.

The dataset, consistent with the NSL-KDD structure, contains 43 columns. These columns encompass 41 distinct features, a 'label' column indicating the type of network connection (e.g., 'normal', 'attack'), and a 'difficulty' column, which is an auxiliary attribute not used for classification. The feature_names list was programmatically applied as column headers to the loaded DataFrame, conditional on the DataFrame's column count matching the expected 43, as illustrated in Figure 1.

This foundational step is crucial for accurate data referencing throughout the machine learning pipeline. A sample of the loaded and named data is presented in Figure 2.

2. Binary Label Creation
Following the successful loading and initial naming of the dataset, the next crucial step involved transforming the multi-class 'label' column into a binary target variable. The original 'label' column specifies various network connection types, including 'normal' and numerous attack categories (e.g., 'neptune', 'warezclient', 'portsweep'). For a binary classification task, these attack types need to be unified into a single 'attack' class.

To achieve this, a new column, binary_label, was engineered. This column assigns a value of 0 to connections labeled as 'normal' and 1 to all other connections, effectively consolidating all attack types under a single 'attack' class. This binarization simplifies the classification problem, allowing the Logistic Regression model to distinguish between legitimate network traffic and malicious activities.

3. Model Evaluation
Following the grid search for optimal hyperparameters, the performance of the best Logistic Regression model was rigorously evaluated on the unseen test dataset. Key classification metrics, including the Classification Report, Confusion Matrix, and ROC AUC score, were computed to assess the model's efficacy in detecting network intrusions. The output of these evaluations is presented in Figure 3.


3.1. Classification Report and Confusion Matrix
The Classification Report provides a detailed breakdown of precision, recall, and F1-score for both classes (0: normal, 1: attack).

Precision (Class 0 - Normal): 0.96

Recall (Class 0 - Normal): 0.97

F1-score (Class 0 - Normal): 0.96

Precision (Class 1 - Attack): 0.97

Recall (Class 1 - Attack): 0.96

F1-score (Class 1 - Attack): 0.96

The overall accuracy of the model was 0.96, with a weighted average F1-score of 0.96, indicating strong performance across both classes.

The Confusion Matrix further elaborates on the model's predictions:

True Negatives (TN): 150 (correctly predicted normal connections)

False Positives (FP): 5 (normal connections incorrectly predicted as attack)

False Negatives (FN): 5 (attack connections incorrectly predicted as normal)

True Positives (TP): 142 (correctly predicted attack connections)

The low numbers of False Positives and False Negatives demonstrate the model's capability to accurately distinguish between normal and intrusive network behaviors, with minimal misclassifications in both directions.

3.2. Receiver Operating Characteristic (ROC) Curve and AUC
The Receiver Operating Characteristic (ROC) curve visually represents the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR) at various classification thresholds. The Area Under the Curve (AUC) quantifies the overall performance of the classifier, ranging from 0.5 (random classification) to 1.0 (perfect classification).

The model achieved an ROC AUC score of 0.98. This exceptionally high value indicates that the Logistic Regression model possesses excellent discriminative power, being highly capable of distinguishing between normal and attack traffic. The ROC curve itself, as depicted in Figure 3, which closely follows the top-left boundary of the plot, further reinforces this finding, illustrating a robust performance across different classification thresholds.

Note: The user warning "Found unknown categories in columns [1] during transform" indicates that during the preprocessing of the test set, the OneHotEncoder encountered categories in the 'protocol_type' column that were not present in the training data. While handle_unknown='ignore' prevented an error by encoding these as all zeros, it is crucial to ensure that the training data is sufficiently representative of all categories that might appear in the test or production environment. This finding suggests a potential need for more comprehensive training data or a more robust strategy for handling novel categories.

4. Advanced Interpretability: Permutation Importance
To understand which transformed features contribute most significantly to the model's predictions, permutation importance was calculated. This method assesses the reduction in model performance (in this case, F1-score) when a single feature's values are randomly shuffled, thereby breaking its relationship with the target variable. A larger drop in performance indicates higher importance. The analysis was performed on the transformed feature space (102 features) that the Logistic Regression model directly utilizes, as shown in Figure 4.

The top 10 most important transformed features, based on their mean permutation importance, are as follows:

num_dst_host_same_src_port_rate: 0.028713

num_dst_host_srv_count: 0.026733

num_count: 0.024422

num_srv_serror_rate: 0.022112

num_same_srv_rate: 0.019882

num_dst_host_srv_diff_host_rate: 0.018812

num_wrong_fragment: 0.017822

cat_service_private: 0.016582

num_serror_rate: 0.014191

num_dst_host_same_srv_rate: 0.008911

Findings:

Host-Related Features Dominant: Several features related to the destination host, such as num_dst_host_same_src_port_rate (rate of connections to the same service from the same source port), num_dst_host_srv_count (count of connections to the same host as the current connection), and num_dst_host_srv_diff_host_rate (rate of connections to different hosts than the current connection), show high importance. This suggests that the behavior patterns of traffic directed towards or originating from specific hosts are crucial indicators for intrusion detection.

Service Interaction Metrics: Features like num_count (number of connections in the past two seconds) and num_same_srv_rate (rate of connections to the same service as the current connection) are also highly influential. These metrics capture the frequency and consistency of service interactions, which can deviate significantly during an attack.

Error Rates: num_srv_serror_rate (rate of SYN errors) and num_serror_rate (rate of connections with SYN errors) indicate that connection establishment errors are strong predictors of malicious activity.

Categorical Feature Importance: The one-hot encoded categorical feature cat_service_private (indicating connections to the 'private' service) also emerged as a significant predictor, highlighting the importance of specific service types in the detection process.

Fragment and Compromise Metrics: num_wrong_fragment (number of 'wrong' fragments) and num_compromised (number of compromised conditions) are also important, directly pointing to unusual packet fragmentation or system integrity breaches.

These findings provide valuable insights into the network characteristics that the Logistic Regression model leverages to identify intrusions, emphasizing the importance of host behavior, service interaction patterns, and error indicators.

5. SHAP Analysis for Local Interpretability
To provide a more granular understanding of individual feature contributions to specific predictions, SHAP (SHapley Additive exPlanations) values were computed for the Logistic Regression model. SHAP values quantify the impact of each feature on the model's output for a given instance, providing both the magnitude and direction (positive or negative impact on the prediction of an attack).

The SHAP summary plot visually aggregates the SHAP values across the test dataset, offering insights into overall feature importance and the distribution of their effects, as depicted in Figure 5.


Key Findings from the SHAP Summary Plot:

Overall Importance: The vertical axis lists features in descending order of their overall impact on the model output (i.e., the magnitude of their SHAP values). This ranking largely aligns with the permutation importance findings, reinforcing the significance of features such as num_dst_host_srv_count, num_same_srv_rate, num_count, and num_srv_serror_rate.

Direction of Impact (Color):

Red dots: Indicate instances where a higher value of the feature leads to a higher model output (i.e., a stronger prediction of 'attack').

Blue dots: Indicate instances where a lower value of the feature leads to a higher model output.

Distribution of Impact (Spread): The horizontal spread of dots for each feature illustrates the range of SHAP values, showing how much that feature impacts different predictions.

Specific Feature Interpretations from SHAP:

num_dst_host_srv_count: Higher values (red dots) for this feature tend to increase the likelihood of an attack prediction, suggesting that a large number of connections to the same service on the destination host is often indicative of malicious activity.

num_same_srv_rate: Higher values (red dots) of this feature also contribute to higher attack predictions. This implies that a high rate of connections to the same service often correlates with an attack.

num_count: Higher values (red dots) are associated with increased attack predictions, suggesting that a high number of connections in a short time frame points to suspicious behavior.

cat_service_private: The distribution of this categorical feature's SHAP values shows its specific impact. A presence of this service (likely 1 for the one-hot encoded feature) can push predictions towards 'attack' or 'normal' depending on other features, indicating its conditional importance.

The SHAP analysis provides a robust framework for explaining the Logistic Regression model's decisions, not just for overall feature relevance but also for their specific influence on individual predictions, which is critical for trust and actionable insights in security applications.
