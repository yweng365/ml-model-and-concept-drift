# Model and Concept Drift

---

**Course Code:** MMF1922HF Data Science  
**Authors:**  
- Ganhui (Lucy) Huang  
- Xinyi (Shanice) Li 
- Jingwen (Vivien) Shao
- Yuting (Yoyo) Weng  

**Date:** October 31st, 2024

---

## Table of Contents
- [Introduction](#Introduction)
- [Drift Detection and Challenges](#Drift_Detection_and_Challenges)
- [Drift Adaptation Techniques](#Drift_Adaptation_Techniques)
- [Real World Example: Financial Risk Prediction (FRP) Model](#Real_World_Example_Financial_Risk_Prediction_(FRP)_Model)
- [Conclusion](#Conclusion)
- [References](#References)

---

## 1. Introduction

Model drift is defined as the observed degradation of machine learning model performance due to changes in data or the relationships between input and output variables. Models based on historical data can soon become stagnant as the world is constantly changing, including new variations, patterns, and trends. These new data points cannot be captured by the existing data. If the machine learning model's training is not aligned with the latest data, it may not accurately interpret the data and forecast effectively. So, model drift can have a negative impact on model performance, leading to poor decision-making and predictions.

### 1.1 Types of Model Drift

The two main categories of model drift are concept drift and data drift. The data drift refers to changes in input features' distributions, whereas the concept drift refers to changes in the relationship between model inputs and outputs. In this project, we will mainly focus on the concept drift.

### 1.2 Types of Concept Drift

There are various types of concept drift patterns, including gradual concept drift, sudden concept drift and recurring concept drift. 
- **Gradual Concept Drift**: Gradual Concept Drift is the most common concept drift. Like its name, the term "drift" refers to a gradual change or movement. Gradual concept drift occurs as the underlying data patterns change over time. In production, we generally see a smooth decay in the core model quality metric over time. The actual speed of decay varies and is greatly influenced by the modeled process and the rate of change in the environment.
- **Sudden Concept Drift**：Sudden concept drift is the opposite of gradual concept drift; it is an abrupt and unexpected change in the model environment. COVID-19 is a good example of a sudden change that impacted machine learning models across industries. For instance, as the national lockdown was announced, the actual sales of loungewear suddenly increased.
- **Recurring Concept Drift**: The "recurring" concept drift refers to repeated or cyclical pattern changes, such as increases in sales during holidays, discounts, or Black Friday.

## 2. Drift Detection and Challenges

### 2.1 Consequences of Concept Drift
Concept drift has several side effects on the model performances. To strat with, one of the main consequences of concept drift is model degradation. Specifically, models trained on past data become less accurate over time because the data points are insufficient to capture the complexity of the problem, plus the system environment is dynamic and progressively subject to changes, making it difficult for a single model to provide accurate predictions.
In addition, concept drift increases the complexity in model maintenance. Precisely, frequent retraining on updated data is essential to maintain model relevance, which is a resource-intensive and computationally expensive process, especially when model and data updates are needed during periods of economic instability.
Moreover, concept drift also increases the likelihood of model overfitting and underfitting since models frequently retrained or adjusted to new data may overfit to short term changes, making the model unable to make accurate predictions in long term.

### 2.2 Concept Drift Detection
There are several methods to capture concept drift.
One of the most-used detection approaches is Statistical Process Control (SPC) method. SPC method involves monitoring model performance and data stability through various techniques including Error Rate Monitoring, ADWIN, and Cumulative Sum (CUSUM) methods track model error rates and data distributions to signal drift. Another category of detection approaches is Window-Based Distribution Methods, which compare incoming data distributions across windows, whereas Adaptive and Ensemble-Based Learning like Ensemble and Incremental Learning adapt models to new data patterns. Moreover, Uncertainty-Based Detection methods, including Uncertainty Drift Detection (UDD), Predictive Entropy, and Confidence Intervals, assess model confidence, flagging drift when uncertainty rises. 
Finally, Hybrid and Complex Data Techniques, such as PCA-Based Detection and Graph-Based Analysis, tackle drift in high-dimensional and networked data. Together, these techniques ensure accurate, up-to-date models by detecting shifts in data distributions.

### 2.3 Challenges in Concept Drift Detection
Identifying concept drift poses several challenges, including differentiating concept drift from random noise, especially in high-frequency data streams such as financial transactions. Drift appears in multiple forms - abrupt, gradual, incremental, and recurring — yet most detection methods often specialize to identify only one type, imposing greater challenges on addressing multiple drift types at the same time. Additionally, high-dimensional and complex data structures make drift detection more complicated, as detecting shifts in such structures demands substantial memory and computational resources.


## 3. Drift Adaptation Techniques

After the detection of drift, we look into the techniques to effectively adapt to model and concept drift. Depending on the nature of the drift, several strategies can be used. We will then introduce some strategies, grouping into four categories ———— general adaptation strategies, deep learning-specific methods,ensemble methods, and evaluation approaches to ensure model resilience.

### 3.1 General Adaptation Strategies

Since machine learning algorithms usually optimize an objective function against a static training set, the idea is to perform modifications to apply them on evolving data. In this paper, wee propose six different modifying strategies. In each of the strategies, the suitable situation is listed.

- **Detect and Increment**: This approach uses a drift detector to monitor model performance continuously and incrementally updates model parameters with new data. The model is updated only if drift is detected, with the
pipelines trained incrementally with the latest s (sliding window) batchesIt is suitable for gradual drift without significant changes to the model configuration.

- **Detect and Retrain**: When drift is detected, the model is retrained from scratch using the most recent data while keeping the original structure intact. This ensures that the model is up-to-date with recent data patterns and works effectively for moderate changes.

- **Detect and Warm-Start**: Upon drift detection, the AutoML process is re-run with a warm start, building on the best previously evaluated configurations. This technique is well-suited for moderate drift scenarios where the existing model needs slight optimization.

- **Detect and Restart**: In the case of major drift, this approach involves re-running the entire AutoML process from scratch, completely re-optimizing the model. This method is computationally intensive but necessary for significant data distribution changes.

- **Periodic Restart**: The AutoML process is periodically re-run at fixed intervals regardless of drift detection. This ensures that the model remains consistently updated, which is useful in highly dynamic environments but can be resource-heavy.

- **Train Once**: As a baseline, the model is trained initially and not updated further. This approach serves as a benchmark for comparing how well adaptive strategies perform against a static model.

### 3.2 Deep Learning-Specific Adaptation Techniques

- **Model Parameter Updating**: This technique involves updating model weights in response to drift while maintaining the same network architecture. It is effective for incremental and gradual drift scenarios.
  - *Fully Parameter Updating*: Updates all model parameters using new data, ensuring complete adaptation but potentially suffering from slow convergence.
  - *Partially Parameter Updating*: Updates only selected parameters, which mitigates catastrophic forgetting and provides faster adaptation.

- **Model Structure Updating**: This approach adjusts the architecture of the neural network to handle drift more effectively. It is particularly useful for addressing complex or multiple types of drift.
  - *Network Width Adjusting*: Adds units or branches to the existing network structure, allowing the model to adapt incrementally to new data distributions.
  - *Network Depth Adjusting*: Adds layers to the network, enhancing adaptability to abrupt and recurrent drift.

### 3.3 Ensemble Methods for Drift Adaptation

- **Adaptive and Dynamic Ensemble (ADE-SVM)**: This method involves an ensemble of support vector machines that adapts over time to account for new data batches and changing financial distress concepts. The model adapts to drift by:
  - *Incrementally Updating Base SVMs*: The candidate SVMs are incrementally updated using the latest data batches, ensuring the ensemble remains relevant for financial risk prediction as data changes.
  - *Dynamic Selection of SVMs*: Base SVMs are adaptively selected based on their predictive ability and classifier diversity. This helps maintain an optimal combination of models to handle the evolving data distributions.
  - *Time Window Approach*: A time window is applied to determine the relevance of new data batches, and older data that may no longer be informative is excluded. This ensures that only the most recent, relevant data influences the model's decision-making.
  - *Dynamic Weighting*: Base SVMs are dynamically weighted according to their validation performance on the latest data, which enhances the ensemble's adaptability and predictive accuracy in real-world settings.

- **Super-Ensemble Methods**: Super-ensemble techniques involve combining multiple models to create a robust prediction. These ensembles adapt to drift by dynamically updating the weights of individual models or by assimilating new data continuously. The idea is to leverage diverse model strengths to improve resilience to various types of drift, including surface drift prediction and concept drift.

### 3.4 Evaluating and Adapting Resistance to Model Drift

Evaluating and adapting model resistance to drift is crucial to ensure robustness, especially in high-stakes environments. Adaptation strategies should also consider the model's ability to defend against adversarial influences and maintain stability under evolving conditions.

- **Adversarial Drift and Adaptation**: Machine learning models in sensitive environments can be vulnerable to adversarial drift, where crafted data points are introduced to induce incorrect classifications. To adapt to such drifts, models can be enhanced with techniques such as adversarial training, which involves training the model on adversarial examples to improve robustness. This is particularly important for domains such as intrusion detection systems (IDS) and fraud detection.

- **Monte Carlo Simulation for Drift Evaluation**: Monte Carlo simulations can be utilized not only to evaluate susceptibility to drift but also to test different adaptive strategies under varying conditions. By repeatedly sampling and testing configurations, it is possible to identify optimal adaptation methods and enhance model resilience against drift, including adversarial scenarios.

- **Adaptive Evaluation Techniques**: Evaluating model resistance to drift requires adaptive approaches such as:
  - *Centroid Anomaly Detection*: Continuously adjusts centroids based on new data to detect anomalous shifts in distribution, providing an early indication of drift.
  - *Adaptive SVM and HMM*: Support vector machines and hidden Markov models can be adapted to enhance resilience to concept drift. Adaptive SVMs use incremental learning to update support vectors with new information, while HMMs leverage ensemble methods to improve stability in dynamic settings. These techniques are particularly valuable for maintaining model performance in scenarios with evolving data distributions.

## 4. Real World Example: Financial Risk Prediction (FRP) Model

### 4.1 Problem Description
A Financial Risk Prediction (FRP) model is a predictive model used to assess and predict the likelihood of financial risks, such as defaults, operational losses, bankruptcy, across various financial sectors. While financial market is a non-stationary environment where the constantly changing patterns and characteristics of the market often leads to concept drift when fitting new batches of financial data into the predictive model, and thus leading to poor alignment with future market conditions. As a result, due to this ongoing change, FRP models may struggle to maintain accuracy and reliability when predicting upcoming risks. Therefore, it is imporant to develop effective strategies for addressing concept drift, enhancing the model’s adaptability and predictive performance in the face of evolving market conditions.

### 4.2 Traditional Approaches to Address Concept Drift in FRP Model
Traditional FRP models utilize windows and batch selection to address concept drift, while these methods are still flawed, making the models incapable of predicting future market situations accurately in all cases. Some examples of traditional methods and their drawbacks are listed below:
- Adaptive sliding time window (ASTW): Adjusts the window size dynamically based on the recent data but may not effectively handle rapid changes.
- Full Sliding Time Window (FSTW): Considers all available data batches, leading to models that may become outdated as new data emerges.
- Batch Selection (BS): Selects data batches similar to the most recent batch for training, but its effectiveness decreases with increased concept drift.

### 4.3 Adaptive and Dynamic Ensemble of Support Vector Machines (ADE-SVM) Approach
ADE-SVM is a novel approach in resolving issues relating to concept drift in dynamic financial environment. Specifically, features of ADE-SVM Approach such as incremental construction of candidate SVMs, adaptive selection of base SVM and dynamic combination mechanism allow the predictive model adapt to new data batches that emerge over time, ensuring the prediction model remains relevant and accurate. Specific explanation of each feature is listed below:
- Incremental construction of candidate SVMs: Each new batch of data forms a candidate model, and the system runs the SVM algorism on different combinations of data to form candidate SVMs, allowing the model to learn from both historical and recent patterns.
- Adaptive selection of base SVMs: System selects base SVM according to the performance of candidate models on the recent validation dataset, ensuring the pertinence and diversity of the model for future predictions.
- Dynamic combination mechanism: Selected base SVMs are weighted dynamically, and the outputs are combined through a weighted voting system. This mechanism ensures that the model that is most consistent with the current financial situation has a greater impact on the forecast.

### 4.4 Empirical Results & Comparisons
An analysis was conducted by applying the ADE-SVM model and various traditional Financial Risk Prediction (FRP) models to financial data from several Chinese companies over the period 2000 to 2008. When comparing the model outputs with empirical data, the ADE-SVM model demonstrated significantly lower error rates, greater stability, and improved dynamic adaptability compared to traditional models.

## 5. Conclusion
Overall, in this project, we introduced the causes and impacts of model drift and concept drift in models. Then we discussed several approaches to detect concept drift including Statistical Process Control (SPC) Methods, Window-Based Distribution Methods, Adaptive and Ensemble-Based Learning Methods, Uncertainty-Based Detection and Hybrid & Complex Data Techniques. Probing into methods aiming to address concept drift issues in models, we introduced 4 specific categories of drift adaptation techniques, namely general adaptation strategies, deep learning-specific methods, ensemble methods, and evaluation approaches. In addition, a real-world case of FRP model and ADE-SVM approach is presented to show the applications of drift detection and adaptation techniques in financial markets.

## 6. References

Bayram, F., Ahmed, B. S., & Kassler, A. (2022). From concept drift to model degradation: An overview on performance-aware Drift Detectors. Knowledge-Based Systems, 245, 108632. https://doi.org/10.1016/j.knosys.2022.108632 

Holdsworth, J., Stryker, C., & Belcic, I. (2024, July 16). What is model drift?. IBM. https://www.ibm.com/topics/model-drift 

Machine Learning Monitoring, part 5: Why you should care about data and concept drift. Evidently AI - Open-Source ML Monitoring and Observability. (n.d.-a). https://www.evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift#concept-drift 

What is concept drift in ML, and how to detect and address it. Evidently AI - Open-Source ML Monitoring and Observability. (n.d.-b). https://www.evidentlyai.com/ml-in-production/concept-drift#types-of-concept-drift 

Sun, J., Li, H., & Adeli, H. (2013). Concept drift-oriented adaptive and dynamic support vector machine ensemble with time window in corporate financial risk prediction. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 43(4), 801-813.

