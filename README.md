> **Note:** GitHub often fails to render large Jupyter Notebooks. To view the full project with all plots and training results, please use the DagsHub link below:
>
> [![View on DagsHub](https://dagshub.com/static/badge.svg)](https://dagshub.com/Chooladeva/Diabetic-Retinopathy-Classification)

# Diabetic Retinopathy Severity Grading: Custom CNN vs. Transfer Learning

## Exploratory Data Analysis & Preprocessing

Before any model was built, a thorough Exploratory Data Analysis (EDA) was conducted to understand the structure, quality, and challenges of the dataset. The findings directly shaped the preprocessing pipeline and model design decisions that followed. This document summarises the key observations from the EDA and the preprocessing steps applied to prepare the data for training.Dataset DescriptionThe dataset consists of high-resolution retinal fundus photographs collected under varying imaging conditions. Each patient contributes two images — one for the left eye and one for the right eye — with filenames in the format {patient_id}_left.jpeg and {patient_id}_right.jpeg. Each image has been clinically graded on a five-level diabetic retinopathy severity scale:

- 0- No DR (Healthy)
- 1- Mild DR
- 2- Moderate DR
- 3- Severe DR
- 4- Proliferative DR

### Exploratory Data Analysis

**I. Class Distribution Analysis**

The first and most critical finding from the EDA is the severe class imbalance in the dataset. The bar chart analysis revealed that Grade 0 (healthy) accounts for the overwhelming majority of images, while the clinically important severe and proliferative grades are represented by far fewer samples.

| Grade | Description   | Count               |
| ----- | ------------- | ------------------- |
| 0     | Healthy       | Majority (~73%)     |
| 1     | Mild          | Small minority      |
| 2     | Moderate      | Moderate minority   |
| 3     | Severe        | Very small minority |
| 4     | Proliferative | Smallest group      |

This imbalance is not just a statistical concern — it has direct clinical implications. A model that ignores the imbalance would learn to predict healthy for most images and achieve superficially high accuracy while failing to detect the disease cases that matter most. This finding motivated the use of weighted sampling, Focal Loss, and Mixup augmentation in both models.

**II. Symmetry Analysis — Left vs. Right Eye**

Since each patient contributes both a left and right eye image, it was important to understand how consistent DR severity grades are between the two eyes. This has two implications — clinical and methodological.

Key findings:

- 87.25% of patients have the same DR grade in both eyes, indicating that the disease typically progresses similarly in both eyes
- Grade 0 symmetry: ~94% — if one eye is healthy, the other almost certainly is too
- Grade 3 and 4 symmetry: ~75% — advanced disease tends to affect both eyes, though some asymmetry exists
- Grade 1 symmetry: ~50% — the lowest symmetry, with many cases where one eye is mild while the other is still healthy (36% of Grade 1 cases)

The low symmetry in Grade 1 is clinically meaningful — it suggests that mild DR often represents the earliest observable stage of the disease, where one eye has just begun to show signs while the other has not yet progressed. This also explains why Grade 1 is the hardest grade to classify reliably.

Methodological implication: The high inter-eye correlation (87.25%) meant that naive random splitting of images into train and test sets would cause data leakage — a patient's left eye image could appear in training while their right eye appears in testing. This was addressed through patient-level data splitting, described in the preprocessing section below.

**III. Brightness Distribution by Class**

The brightness analysis examined whether DR severity correlates with image brightness — which would suggest that disease progression affects how images are captured or appear visually.

Key findings:

- Brightness levels are largely consistent across all five DR grades — the central 50% of brightness values overlap significantly between classes
- Grade 0 shows the most outliers — some healthy eye images appear excessively bright or washed out, likely due to imaging artifacts
- There is wide within-class brightness variation across all grades, reflecting the diverse imaging conditions under which the dataset was collected

The consistent brightness across grades confirms that the model cannot rely on overall image brightness as a shortcut for grading — it must learn actual pathological features. The wide brightness variation motivated the inclusion of colour jitter augmentation (±0.2 brightness and contrast) to make both models robust to these real-world imaging inconsistencies.

**IV. Image Resolution and Shape Analysis**

Understanding the resolution and shape characteristics of the raw images was essential for designing the preprocessing pipeline.

Key findings:

- No images are perfectly square — all images are rectangular with a landscape (wider than tall) orientation
- Average aspect ratio: ~1.46 — images are noticeably wider than tall on average
- Large resolution variation — some images are as small as 400×300 pixels while others exceed 5,000 pixels in width
- Small images may lack the fine detail needed to identify subtle lesions (microaneurysms, small haemorrhages)
- Very large images carry more detail but are computationally expensive to process

These findings made it clear that a structured normalisation pipeline was essential — directly resizing the raw rectangular images to a square input format would introduce distortion, and the large resolution variation would make lesion sizes inconsistent between images even within the same DR grade.

### Data Preprocessing

The EDA findings motivated a five-stage preprocessing pipeline designed to standardise image quality, size, and retinal scale before any model training.

**Image Preprocessing Pipeline**

- **Stage I — Automated Cropping and Noise Removal**

Many raw images contained large black borders around the retinal region, along with edge artifacts that carry no clinical information. An automated cropping step detected the central retinal region and removed these borders, ensuring the model focuses exclusively on relevant content from the first pixel.

- **Stage II — Standardised Radius Scaling**

Because images were captured at varying distances, the apparent size of the retina varies significantly between images — a lesion that looks large in one image may look tiny in another despite representing the same clinical severity. To fix this, a radius-based scaling method was applied:

- The radius of the retina was estimated from the middle row of pixels in each image
- All images were resized so that the retinal radius was standardised to 384 pixels

This ensures that anatomical structures (the optic disc, blood vessels, lesion regions) appear at a consistent scale across all images, regardless of the original imaging distance.

- **Stage III — Square Padding**

After radius scaling, images were placed onto a square canvas with a neutral grey (128) background. This step preserves the original proportions of the retina — avoiding the distortion that direct rectangular-to-square resizing would introduce — while producing a uniform square input size ready for the neural network.

- **Stage IV — Ben Graham Contrast Enhancement**

The dataset included images with varying visual quality — haze, uneven illumination, and colour inconsistencies that make subtle features like microaneurysms and small haemorrhages difficult to see. Ben Graham's contrast enhancement method was applied:

Enhanced = Image - GaussianBlur(Image) + 128

This subtracts a blurred version of the image from the original, which effectively removes background illumination gradients and makes fine structures such as blood vessels, lesions, and exudates stand out more clearly against the retinal background.

- **Stage V — Circular Masking**

After contrast enhancement, some edge artifacts became more pronounced at the image borders. A circular mask was applied to retain only the central retinal region while filling the outer areas with a neutral grey background. This ensures the model attends exclusively to clinically relevant regions and is not confused by processing artifacts at the image edges.

Preprocessing progression:

```
Raw Image  
↓  
Automated Cropping (removes black borders)  
↓  
Radius Scaling (standardises retinal size to 384px radius)  
↓  
Square Padding (grey canvas, preserves aspect ratio)  
↓  
Ben Graham Contrast Enhancement (improves lesion visibility)  
↓  
Circular Masking (focuses on retinal region only)  
↓  
Preprocessed Image (ready for model training)
```

**Data Splitting**

The symmetry analysis finding — that 87.25% of patients have matching grades in both eyes — made conventional random image splitting inappropriate. If both eyes from the same patient were split across training and test sets, the model would effectively be evaluated on patients it had already seen, producing artificially optimistic performance estimates.

A patient-level splitting strategy was applied instead:

- Both images from the same patient are always assigned to the same split (either both in training or both in testing)
- This completely eliminates patient overlap between training and test sets
- The dataset was split 80% training / 20% testing

Final split distribution:

| Grade     | Description     | Train Count | Test Count |
| --------- | --------------- | ----------: | ---------: |
| 0         | No DR (Healthy) |      20,652 |      5,158 |
| 1         | Mild            |       1,940 |        503 |
| 2         | Moderate        |       4,244 |      1,048 |
| 3         | Severe          |         697 |        176 |
| 4         | Proliferative   |         567 |        141 |
| **Total** |                 |  **28,100** |  **7,026** |

**Data Augmentation and Normalisation**

To further improve model generalisation and address the class imbalance identified in the EDA, a targeted augmentation strategy was applied dynamically during training (on-the-fly) rather than as a fixed offline preprocessing step. This means the model sees a different variation of each image on every epoch, effectively expanding the training set without storing additional data.

Augmentation strategy by class:

| Grade          | Augmentations Applied                                                                                    |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| 0 (Healthy)    | Random horizontal flip only                                                                              |
| 1–4 (Diseased) | Horizontal flip + Random rotation (0°, 90°, 120°, 180°, 270°) + Colour jitter (±0.2 brightness/contrast) |

Minority classes receive more aggressive augmentation because they have fewer unique training images and are at higher risk of being memorised by the model rather than generalised from.
After augmentation, all images were:

- Converted to PyTorch tensors — pixel values scaled from [0, 255] to [0, 1]
- Normalised using ImageNet statistics — mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225] per channel


ImageNet normalisation is applied even to RetinalNet (trained from scratch) because the statistics provide a reasonable and consistent input distribution given that retinal images are captured under visible-light conditions similar to natural photographs.


## Model Evaluation & Results

This document covers the complete evaluation of both models — EfficientNet-B4 (transfer learning) and RetinalNet (custom CNN trained from scratch) — on the diabetic retinopathy grading task. Evaluation goes beyond standard metrics to include interpretability analysis, calibration assessment, and a real-world prevalence alignment test on a large unlabelled dataset of approximately 53,000 retinal images.

### Evaluation Metrics Used

| Metric                                 | Why It Was Used                                                                                                                                                  |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| QWK (Quadratic Weighted Kappa)         | Primary metric — accounts for ordinal class ordering and class imbalance. Predicting Grade 4 as Grade 0 is penalised far more than predicting Grade 1 as Grade 0 |
| Accuracy                               | Reported but used cautiously — misleading due to 73.5% Grade 0 prevalence                                                                                        |
| MCC (Matthews Correlation Coefficient) | Reliable single-number summary for imbalanced multiclass problems                                                                                                |
| Precision / Recall / F1                | Per-class breakdown to identify where each model succeeds and fails                                                                                              |
| ROC AUC                                | General screening ability — distinguishing diseased from healthy                                                                                                 |
| Average Precision (AP)                 | PR curve summary — more informative than ROC for imbalanced data                                                                                                 |
| Reliability Diagrams                   | Model calibration — how well predicted probabilities reflect true outcomes                                                                                       |

### Overall Results

RetinalNet outperforms EfficientNet-B4 in 6 out of 9 evaluation categories. However, EfficientNet retains a critical advantage on clinically important metrics — particularly Grade 4 recall, which directly determines how many of the most severe, vision-threatening cases are caught.

| Metric          | EfficientNet-B4 |      RetinalNet | Better Model    |
| --------------- | --------------: | --------------: | --------------- |
| QWK             |           0.611 |          0.6399 | RetinalNet      |
| Accuracy        |           73.7% |           76.5% | RetinalNet      |
| MCC             |          0.3328 |          0.3856 | RetinalNet      |
| ROC AUC         |          0.7877 |          0.7719 | EfficientNet-B4 |
| Grade 4 Recall  |            0.74 |            0.60 | EfficientNet-B4 |
| Grade 2 AP      |            0.44 |            0.55 | RetinalNet      |
| Calibration     |   Overconfident | Well-calibrated | RetinalNet      |
| Grade 1 Recall  |            0.11 |            0.04 | EfficientNet-B4 |
| Inference Speed |    Slower (TTA) |          Faster | RetinalNet      |

#### Per-Class Performance

**EfficientNet-B4 (with Test-Time Augmentation)**

| Grade             | Precision | Recall |   F1 | Support |
| ----------------- | --------: | -----: | ---: | ------: |
| 0 — Healthy       |      0.82 |   0.91 | 0.86 |   5,158 |
| 1 — Mild          |      0.13 |   0.11 | 0.12 |     503 |
| 2 — Moderate      |      0.58 |   0.21 | 0.31 |   1,048 |
| 3 — Severe        |      0.40 |   0.57 | 0.47 |     176 |
| 4 — Proliferative |      0.41 |   0.74 | 0.52 |     141 |

**RetinalNet (Standard Single-Pass Inference)**

| Grade             | Precision | Recall |   F1 | Support |
| ----------------- | --------: | -----: | ---: | ------: |
| 0 — Healthy       |      0.83 |   0.94 | 0.88 |   5,158 |
| 1 — Mild          |      0.10 |   0.04 | 0.05 |     503 |
| 2 — Moderate      |      0.71 |   0.28 | 0.40 |   1,048 |
| 3 — Severe        |      0.30 |   0.61 | 0.40 |     176 |
| 4 — Proliferative |      0.50 |   0.60 | 0.55 |     141 |

**Grade-by-Grade Interpretation**

- Grade 0 — Healthy

Both models perform well, as expected given the abundance of training data. RetinalNet has slightly higher recall (0.94 vs 0.91), meaning it correctly identifies more healthy eyes. However this also means it is slightly more likely to call a diseased eye healthy.

- Grade 1 — Mild DR

The weakest class for both models. EfficientNet catches 11% of mild cases while RetinalNet catches only 4%. Both are clinically concerning — mild DR is the earliest detectable stage and the optimal point for intervention. The failure here is primarily due to visual similarity with healthy eyes and limited training data.

- Grade 2 — Moderate DR

RetinalNet is meaningfully better — precision of 0.71 versus EfficientNet's 0.58. When RetinalNet predicts moderate DR, it is correct 71% of the time. Both models still have low recall (0.21 and 0.28), missing most moderate cases.

- Grade 3 — Severe DR

EfficientNet has higher precision (0.40 vs 0.30) while RetinalNet has marginally higher recall (0.61 vs 0.57). EfficientNet's predictions are more trustworthy, but RetinalNet catches slightly more true cases. Both models perform relatively well here because severe DR has visually distinctive features.

- Grade 4 — Proliferative DR

EfficientNet's recall of 0.74 is its most clinically valuable result — it catches nearly three quarters of the most urgent cases. RetinalNet's 0.60 recall, while respectable, misses 40% of these critical patients. From a deployment standpoint, EfficientNet's higher sensitivity for the most severe grade is a significant clinical advantage.

**Test-Time Augmentation (TTA)**

TTA was applied to EfficientNet-B4 during final evaluation. Each test image is passed through the model in four orientations — original, horizontal flip, vertical flip, and 90° rotation — and the probability distributions are averaged before the final class is selected.

Why TTA was not applied to RetinalNet:

TTA was tested on RetinalNet but decreased performance. EfficientNet-B4, trained with ImageNet pretraining, produces consistent probability estimates across augmented views because its features are orientation-stable. RetinalNet, trained from scratch on a smaller domain-specific dataset, is more sensitive to image orientation and produces inconsistent outputs across views. Averaging inconsistent distributions degrades rather than improves predictions.

#### Model Interpretability — Grad-CAM Analysis

Gradient-weighted Class Activation Mapping (Grad-CAM) was used to visualise which regions of each retinal image each model focused on when making its prediction. This provides insight into whether the models are attending to clinically meaningful areas or being distracted by irrelevant features.

**EfficientNet-B4**

- Shows fine-grained, localised attention — focuses on small specific regions rather than broad areas
- For correct predictions in Grades 1 and 3, attention concentrates on small lesion regions, reflecting the pretrained backbone's ability to detect subtle textures
- Failure mode: In some incorrect predictions, attention shifts to image edges and borders rather than the retinal region, suggesting the model can be distracted by imaging artifacts
- For Grade 4, attention spreads across multiple retinal regions, consistent with the widespread abnormalities characteristic of proliferative DR

**RetinalNet**

- Shows broader, more structured attention — often forms circular or halo-like patterns centred on the retinal region
- For correct predictions in Grades 3 and 4, attention concentrates on the central retina and major blood vessels — clinically appropriate regions where haemorrhages are typically found
- Failure mode: Some incorrect predictions display horizontal striping patterns in the heatmaps, indicating a spatial bias likely introduced during preprocessing (cropping and padding). In Grade 2
- cases misclassified as Grade 0, attention shifts to image edges rather than the central retinal region, causing the model to miss pathological features

**Key insight: Both models are learning meaningful retinal patterns rather than random noise. EfficientNet is better at detecting fine local features but more vulnerable to edge artifacts. RetinalNet captures broader retinal structure better but can be affected by preprocessing-induced spatial bias.**

#### Model Calibration — Reliability Diagram Analysis

Reliability diagrams measure how well a model's predicted probabilities reflect actual outcomes. A perfectly calibrated model's confidence curve follows the diagonal exactly — if it predicts 70% probability, the true frequency should be 70%.

**EfficientNet-B4**

- The reliability curve lies below the diagonal across the lower probability range (0.0–0.5)
- This indicates overconfidence — the model assigns higher probabilities than the actual observed frequencies
- Example: when EfficientNet predicts 40% probability of disease, the true occurrence may be closer to 30%
- Clinically, this can lead to overestimation of risk in borderline cases and more false positives

**RetinalNet**

- The reliability curve closely follows the diagonal across most probability ranges
- This indicates strong calibration — predicted probabilities align well with actual outcomes
- When RetinalNet assigns a confidence score to a prediction, that score is genuinely informative
- Clinically valuable — doctors can rely on the model's probability outputs when making decisions

**Why RetinalNet is better calibrated: Both models use the same label smoothing (0.05), but RetinalNet's domain-specific from-scratch training, combined with its two-stage dropout and BatchNorm in the classifier head, naturally produces better-calibrated outputs. EfficientNet's stronger pretrained features can sometimes produce overconfident predictions that label smoothing alone cannot fully counteract.**

#### Prevalence Alignment Analysis

Both models were used to generate predictions on a large unlabelled dataset of approximately 53,000 retinal images (~53GB). The predicted class distributions were compared against the known training distribution to test whether the models behave realistically on naturally distributed real-world data.

**What Was Measured:** A well-calibrated model should produce a prediction distribution that roughly mirrors the true population prevalence. Significant deviation — particularly over-predicting the majority class — indicates inference-time bias.

**Results:**

| Grade             | Training Prevalence | EfficientNet Predicted | RetinalNet Predicted |
| ----------------- | ------------------: | ---------------------: | -------------------: |
| 0 — Healthy       |               73.5% |         88.8% (+15.3%) |       83.6% (+10.1%) |
| 1 — Mild          |                6.9% |           1.2% (−5.7%) |         2.7% (−4.2%) |
| 2 — Moderate      |               15.1% |          3.0% (−12.1%) |         6.1% (−9.0%) |
| 3 — Severe        |                2.5% |                   3.7% |                 4.9% |
| 4 — Proliferative |                2.0% |                   3.2% |                 2.7% |

**Key Observations**

- Both models over-predict Grade 0 significantly above its true prevalence — EfficientNet by 15.3 percentage points and RetinalNet by 10.1 points. This is not a design failure. It is a well-documented and expected consequence of training on severely imbalanced DR datasets, confirmed by published research using the same EyePACS data.
- Grade 1 nearly disappears from both prediction distributions — dropping from 6.9% true prevalence to 1.2% (EfficientNet) and 2.7% (RetinalNet). This is consistent with published findings that Grade 1 is the most severely under-predicted class across DR models trained on imbalanced data, due to its visual similarity to Grade 0 and limited training examples.
- RetinalNet shows less bias — its Grade 0 over-prediction is 5.2 percentage points lower than EfficientNet, and its Grade 2 predictions (6.1%) are approximately double EfficientNet's (3.0%), which directly explains RetinalNet's superior Grade 2 precision and AP scores.

**Why Training Techniques Cannot Fully Eliminate This Bias**

The weighted sampler creates balanced batches during training but does not change the true data distribution the model encounters at inference time. When the model sees real-world data where 73.5% of images are genuinely Grade 0, its learned prior toward the majority class reasserts itself regardless of the training-time balancing strategies applied.

This phenomenon is formally known as label shift or prevalence shift in the medical imaging literature. Research by Godau et al. (2025) confirms that addressing it requires prevalence-aware recalibration at inference time — adjusting predicted probabilities to better reflect the true population distribution — rather than training-level interventions alone.

**Clinical Implications**

In a real screening scenario:

| Model           | Patients Flagged for Review | Estimated DR Cases Missed |
| --------------- | --------------------------: | ------------------------: |
| EfficientNet-B4 |        ~11.2% of population |     ~57% of true DR cases |
| RetinalNet      |        ~16.4% of population |     ~38% of true DR cases |
| Ideal target    |        ~26.5% of population |     <10% of true DR cases |

Neither model is yet suitable as a standalone clinical screening tool, but both demonstrate genuine learning — QWK scores well above chance — and could function effectively as a triage system to prioritise the most urgent cases for specialist review.

**Which Model to Use**

The right choice depends on the clinical objective:

Choose EfficientNet-B4 if:

- The priority is catching the most severe cases (Grade 4 recall: 0.74)
- Deploying as a screening tool where missing severe disease is the primary risk
- Higher sensitivity is preferred even at the cost of more false positives

Choose RetinalNet if:

- The priority is reliable probability estimates for clinical decision-making (better calibration)
- Identifying moderate DR (Grade 2) with high precision is important
- Faster inference without TTA is required
- A more interpretable, domain-specific model is preferred

**Ideal approach — Ensemble both models: Averaging the probability outputs of both models before making a final prediction would likely outperform either individually. The two models make systematically different errors — EfficientNet is better at the severity extremes, RetinalNet is better at intermediate grades — meaning their errors are unlikely to be highly correlated. An ensemble would produce a more balanced and robust prediction profile across all five DR grades.**

**Shared Limitations**

Both models share limitations that should be considered before any deployment:

| Limitation                       | Detail                                                                            |
| -------------------------------- | --------------------------------------------------------------------------------- |
| Grade 1 detection failure        | Both models miss most mild DR cases (recall: 0.11 and 0.04)                       |
| Training-test QWK gap            | Indicates overfitting — partly from weighted sampling creating distribution shift |
| Majority class bias at inference | Both over-predict Grade 0 on real-world data                                      |
| Grade 2 low recall               | Both miss majority of moderate DR cases                                           |
| Not clinically validated         | Results are from a research dataset, not a validated clinical trial               |

#### Recommendations for Improvement

| Recommendation                    | Expected Impact                                                   |
| --------------------------------- | ----------------------------------------------------------------- |
| CLAHE preprocessing               | Improve Grade 1 detection through better microaneurysm visibility |
| Ensemble both models              | 3–5 point QWK improvement from complementary error profiles       |
| Prevalence-aware recalibration    | Reduce majority class bias at inference time                      |
| DR-specific pretraining           | Largest potential improvement — close the gap toward 0.90 QWK     |
| Ordinal loss function             | Directly aligned with QWK optimisation                            |
| Two-stage classification pipeline | Separate healthy vs. any DR detection from severity grading       |
| Tighter retinal cropping          | Address edge artifact failure modes in both models                |
