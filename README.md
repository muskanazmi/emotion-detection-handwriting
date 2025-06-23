# Emotion-Detection Through Handwriting
Emotion detection using handwriting on EMOTHAW dataset  
This project explores the prediction of Depression, Anxiety, and Stress levels using handwriting dynamics captured through stylus input in .svc files. The model leverages machine learning techniques to analyze temporal, spatial, and statistical handwriting features correlated with the DASS (Depression Anxiety Stress Scales) scores provided for each subject.

#🔧 Steps Performed
✅ 1. Data Loading & Preprocessing
Parsed .svc files and .xls metadata

Extracted columns:
```
x_pos, y_pos, time_stamp, pen_status, azimuth_angle, altitude_angle, pressure
```
Merged DASS scores with each user and session

# 2. Dataset
The EmoThaw dataset captures a wide range of emotion-related handwriting behaviors by recording pen-based input from subjects across several carefully designed handwriting tasks. These tasks are meant to assess fine motor control, cognitive function, and emotional state manifestation through writing.

🧪 Tasks Performed by Subjects
Each subject performed the following structured tasks:

| Task Description             | Writing Style       | Purpose                                              |
|-----------------------------|---------------------|-------------------------------------------------------|
| 🏠 House Drawing             | Freeform Drawing    | Spatial understanding, attention                     |
| 🔷 Pentagon Copying          | Geometric Copy      | Visuospatial coordination, symmetry accuracy         |
| 🕒 Clock Drawing             | Symbolic Drawing    | Cognitive and motor planning                         |
| 🔴 Circle Drawing            | Repetitive Shapes   | Smooth motor control and regularity check            |
| ✋ Handprint Word Copying    | Block Letters       | Focus, grip pressure, hand-eye coordination          |
| ✍️ Cursive Sentence Copying | Cursive Writing     | Fluency, writing rhythm, and stylistic traits         |

🧾 What’s Recorded
Each handwriting session was recorded using a digitizing tablet, producing .svc files for every task.

Every record consists of the following dynamic features:

| Feature         | Description                                             |
|----------------|----------------------------------------------------------|
| `x_pos`        | X-coordinate of the pen on the tablet surface            |
| `y_pos`        | Y-coordinate of the pen on the tablet surface            |
| `time_stamp`   | Time (in milliseconds) since start of recording          |
| `pen_status`   | 1 = Pen on paper, 0 = Pen in air                         |
| `pressure`     | Pressure exerted by the pen                              |
| `azimuth_angle`| Horizontal pen direction (orientation angle)             |
| `altitude_angle`| Vertical elevation of the pen from the tablet surface   |


These detailed time-series features allow us to extract in-air duration, on-paper writing time, number of strokes, total task time, and much more.

# 📊 3. Task Visualization Examples
We visualized these tasks to gain insight into each subject’s writing behavior.

Example (Clock Drawing):
```
plt.plot(df['x_pos'], df['y_pos'], color='blue')
plt.title("Clock Drawing – Subject 1")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.gca().invert_yaxis()
plt.show()
```
✅ Visual cues such as irregular circles, distorted clocks, or shaky pentagons helped identify potential indicators of stress or cognitive load.

Histograms: Visualized score distributions for Depression, Anxiety, and Stress
Correlation Analysis:
Generated bar plots for correlation of features with each emotion
Visualized heatmaps for univariate feature-emotion relationships

🧠 4. Feature Engineering
Extracted key features from stylus movement data:

Air Time (seconds): Time pen is not touching the surface

Paper Time (seconds): Time pen is touching the surface

Total Task Time (seconds)

Number of On-Paper Strokes

These features were saved into features.csv and merged with the DASS metadata for modeling.

🧠5.  Model Training & Evaluation
All training and evaluation code for classification models and dimensionality reduction techniques is consolidated in:
```
📄 train.py
```
This script includes:

🎯 Classification Models
Trained Random Forest Classifiers to predict:
Depression levels
Anxiety levels
Stress levels

Evaluation Metrics:
Accuracy
Confusion Matrix
Classification Report

Additional Analysis:
Computed the percentage of affected vs. non-affected individuals in each test set

📉 Dimensionality Reduction
Linear Discriminant Analysis (LDA):
Reduced features into 2 components
Visualized 2D projections colored by emotional classes
Re-trained Random Forest using LDA-transformed features

Principal Component Analysis (PCA):
Standardized features and projected to 2D space
Visualized PC1, PC2, and emotional intensity in 3D
Performed binary classification (e.g., depression > 9) using PCA features

To reproduce results or modify training pipelines, refer to:
```
python train.py
```

## 📚 Citation

If you use the EMOTHAW dataset in your work, please cite the following paper:

```bibtex
@article{likforman2022emothaw,
  title={EMOTHAW: A novel database for emotional state recognition from handwriting},
  author={Likforman-Sulem, Laurence and Esposito, Anna and Faundez-Zanuy, Marcos and Clémen{\c{c}}on, Stephan and Cordasco, Gennaro},
  journal={arXiv preprint arXiv:2202.12245},
  year={2022}
}
