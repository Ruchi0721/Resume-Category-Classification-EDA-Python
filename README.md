# Resume Category Classification using Machine Learning
In this project, I used Python, Pandas, Scikit-learn, and Matplotlib/Seaborn to build an end-to-end NLP pipeline that automatically classifies resumes into job categories. The dataset contains hundreds of resumes across 25 unique job categories such as Data Science, Network Security, Fitness Training, and more — broken down by raw resume text and category labels.

# Data Cleaning
The project begins with cleaning raw resume text, where I:

•	Removed URLs and hyperlinks using re.sub('http\S+', ...).

•	Stripped social media noise — @mentions, #hashtags, and RT/cc tags.

•	Eliminated special characters, punctuation, and non-ASCII characters (emojis, Unicode symbols).

•	Normalized whitespace to produce clean, model-ready text.

# Data Balancing
Since the dataset had unequal resumes per category, I handled class imbalance by:

•	Identifying the largest class size using .value_counts().max().

•	Oversampling all smaller classes to match it using groupby + .sample(replace=True).

•	Shuffling the balanced dataset using .sample(frac=1) to avoid ordering bias.



# Data Exploration
Before modelling, I explored the dataset visually to understand its structure:

•	**Category Distribution:** Plotted a horizontal bar chart using sns.countplot to compare resume counts across all 25 job categories.

•	**Proportion Analysis:** Created a pie chart using matplotlib to show the percentage share of each category.

•	**Raw Resume Inspection:** Previewed raw and cleaned resume text to validate the cleaning pipeline.


# Key Techniques and Methods
To build the full classification pipeline, I used:

• **Label Encoding** — Converted category names to integer labels using LabelEncoder.

•	**TF-IDF Vectorization** — Transformed resume text into numerical feature vectors using TfidfVectorizer with English stop word removal.

•	**Train-Test Split** — Divided data 80/20 using train_test_split with random_state=42 for reproducibility.

•	**OneVsRest Strategy** — Wrapped classifiers with OneVsRestClassifier to handle multi-class classification.

•	**Model Comparison** — Trained and evaluated both KNN and SVM (SVC) classifiers using accuracy_score.

•**Sparse Matrix Conversion** — Used .toarray() to convert TF-IDF sparse.

