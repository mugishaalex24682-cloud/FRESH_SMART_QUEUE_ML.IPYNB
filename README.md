# BCS3101 Capstone — AI-Assisted Smart Queue Management for Hospitals

**Authors:** MUGISHA ALEX (2024/A/KCS/5461/F), AHURIRA LUCKY (2024/A/KCS/5197/G/F)
**Course:** BCS3101 — Basics of Machine Learning
**University:** Kabale University
**Date:** [Today's date]

---

## Overview

This project develops two machine learning components for an AI-assisted
hospital queue management system proposed for hospitals in Kabale, Uganda:

1. **Urgency Classifier** — predicts whether an emergency department patient
   is urgent (needing immediate attention) or not urgent, based on vital signs
   and demographics captured at triage.

2. **Waiting-Time Predictor** — estimates how long a patient will wait before
   being attended to, based on queue and service variables.

Both models are intended as **decision-support tools**, not autonomous
classifiers. All predictions are presented to a healthcare worker for review.

---

## Component 1 — Urgency Classifier

**Dataset:** MIMIC-IV-ED demo (222 ED visits, 100 patients) — PhysioNet.

**Task:** Binary classification. Urgent = Emergency Severity Index (ESI) 1–2;
not urgent = ESI 3–4.

**Features:** temperature, heartrate, resprate, o2sat, sbp, dbp, pain,
gender, race, arrival_transport.

**Models compared:** Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting.

**Final model:** Tuned Random Forest (`n_estimators=200, max_depth=3`).

**Results:**

| Metric | Value |
|--------|-------|
| CV F1 (5-fold, primary) | **0.7344** |
| Test F1 | 0.5238 |
| Test Accuracy | 0.5238 |
| Test Precision | 0.5789 |
| Test Recall | 0.4783 |

**Confusion matrix (test set, 42 patients):**
- True Negatives: 8
- False Positives: 11
- False Negatives: 8
- True Positives: 15

**Saved model:** `models/urgency_model.joblib`

---

## Component 2 — Waiting-Time Predictor

**Dataset:** Synthetic Dataset of Emergency Healthcare Services
(Zenodo, DOI 10.5281/zenodo.14270002). 112 triage records.

**Task:** Regression. Predict waiting time in minutes.

**Features:** Patient Type, Resource Used, Utilization %, Queue Count
Before Processing, hour of arrival.

**Models compared:** Linear Regression, Ridge Regression, Decision Tree
Regressor, Random Forest Regressor.

**Final model:** Tuned Random Forest Regressor
(`n_estimators=200, max_depth=3`). Selected for robustness on the test
set, which contained an out-of-distribution waiting time (110.79 min)
not represented in training.

**Results:**

| Metric | CV (5-fold) | Test Set |
|--------|-------------|----------|
| MAE (min) | 1.93 | 6.31 |
| RMSE (min) | 2.98 | 22.17 |
| R² | **0.9256** | 0.2520 |

**Saved model:** `models/waiting_time_model.joblib`

---

## Repository Structure
