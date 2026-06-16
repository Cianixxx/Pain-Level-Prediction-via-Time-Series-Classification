Pain Level Prediction via Time-Series Classification (AN2DL) 

This repository contains the codebase for the first Challenge of the Artificial Neural Networks and Deep Learning (AN2DL) course at Politecnico di Milano. This project addresses a supervised time-series classification task using deep learning. 

> 📄 **Full Report:** For an in-depth analysis of the exploratory data analysis, architectural choices, and failed experiments, please refer to the [Full Project Report](./AN2DL_Challenges_Report.pdf).

The entire pipeline, including data processing and model training, was developed and executed using Google Colab.
👥 Team

    Daniele Spini

    Luoxi Massimo Yu

    Simone Delbuono

    Giovanni Fiumalbi

🎯 Objective

The main goal of this project is to predict the pain level of each sequence. The possible classification labels are no_pain, low_pain, and high_pain. To ensure a robust assessment of the model's performance, the primary evaluation metric used is the F1 score.
📊 Dataset & Main Challenges

The provided training dataset (pirate_pain_train.csv) has a shape of (105760, 40). Each sample index contains exactly 160 time steps. The features include:

    Sample index and time.

    Four pain-related measurements.

    Three categorical body-part counters.

    Thirty joint-angle measurements.

A major challenge in this task was dealing with severe class imbalance (77.13% no_pain / 14.22% low_pain / 8.47% high_pain).
🛠️ Pipeline and Methodology
1. Preprocessing & Feature Engineering

    Features with low variance and high correlation were removed.

    Min-max normalization was applied to numerical features.

    One-hot encoding was applied to categorical variables.

    To provide richer temporal context, derived statistics such as velocity, acceleration, rolling mean, and standard deviation (computed using a window of size 10) were added to the joint angles.

2. Handling Class Imbalance

    Initial attempts using SMOTE and Gaussian-noise oversampling yielded slightly better results but caused overfitting and generated unrealistic temporal patterns.

    The final adopted solution was a focal loss function with inverse frequency weights. This ensures that classification errors on minority labels result in greater loss values, guiding the model to balance its learning.

3. Model Architecture

    The best single-performing model relies on a CNN + UniLSTM + Attention architecture.

    A 1D convolutional layer was introduced before the RNN layer to extract local features while preserving temporal relationships.

    Bidirectional LSTM and GRU models were generally the best performers, although some non-bidirectional models showed surprisingly strong results after feature engineering.

    The attention layer computes timestep-wise importance weights and produces a context vector that is fed to a final classification layer.

4. Optimization & Regularization

    AdamW was adopted as the optimizer to apply weight decay directly to the model parameters.

    A learning rate scheduler was used to dynamically adjust the learning rate throughout training.

    Gradient clipping was implemented to prevent excessively large gradients from destabilizing learning.

    Label smoothing was used to introduce additional flexibility into the loss function.

🏆 Results & Ensemble

The team adopted an incremental approach. A baseline UniLSTM started with an F1 score of 0.9368. By adding class balancing, feature engineering, a Conv1D layer, and Attention, the score progressively increased to 0.9484.

To maximize overall performance and stability, a two-stage ensembling strategy was adopted. By aggregating the predictions of the three best-performing models via weighted voting, the team achieved a more stable and accurate final output compared to any single model.
