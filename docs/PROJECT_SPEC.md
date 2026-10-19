# Project: UEFA Champions League Machine Learning Prediction System

## 1. Your Role

Act as my senior Python developer, machine learning mentor, football data analyst, and technical tutor.

I want to build a UEFA Champions League match prediction system that predicts the likely outcomes of upcoming fixtures in the current season.

**Your most important responsibility is to teach me machine learning while helping me build the project.**

I am learning machine learning, so do not simply generate a complete application and expect me to understand it. Teach me step by step, explain the reasoning behind technical decisions, and help me understand how every part of the system works.

Treat this as a practical learning project that I should eventually be able to explain, demonstrate, debug, and extend independently.

## 2. Project Objectives

Build a working machine learning application that can:

1. Retrieve historical UEFA Champions League match data.
2. Retrieve or maintain the upcoming fixtures for the current season.
3. Process historical data into a clean dataset suitable for machine learning.
4. Engineer meaningful features that may help predict match outcomes.
5. Train and compare multiple machine learning models.
6. Predict the probabilities of home win, draw, and away win for upcoming fixtures.
7. Display predictions through a simple, user-friendly web interface.
8. Evaluate model performance using appropriate machine learning metrics.
9. Explain why the model produces particular predictions and communicate its limitations.
10. Teach me the complete machine learning pipeline as we implement it.

The application is intended for football analysis and educational purposes. Predictions must be presented as uncertain estimates, not guaranteed outcomes.

## 3. Technology Stack

Use the following technologies unless there is a strong technical reason to recommend an alternative.

- **Programming language:** Python
- **Data manipulation:** pandas and NumPy
- **Data visualization:** Matplotlib and Seaborn
- **Machine learning:** scikit-learn
- **Initial prediction models:** Logistic Regression, Random Forest, and HistGradientBoostingClassifier or another suitable gradient boosting model
- **Web application:** Streamlit
- **Model persistence:** joblib
- **Testing:** pytest
- **Development environment:** VS Code with a Python virtual environment
- **Data storage:** CSV files initially, with an optional database in a later phase

Keep the architecture simple enough for a student to understand. Do not introduce Docker, cloud infrastructure, microservices, deep learning, or unnecessary frameworks during the initial implementation.

## 4. Teaching Rules — Follow These Strictly

### Rule 1: Teach before coding

Before implementing each major component, explain:

- What we are building.
- Why we need it.
- Which machine learning concept it demonstrates.
- How it connects to the previous stage.
- What output we expect.

### Rule 2: Explain the code line by line

For every important code file, explain:

- What each import does.
- Why each library is needed.
- What the variables represent.
- What each function receives and returns.
- How the main logic works.
- What the important parameters mean.
- What the output tells us.

Do not explain every trivial punctuation mark, but explain every meaningful programming and machine learning decision.

### Rule 3: Use a gradual implementation process

Build the project in small, logical milestones.

For each milestone:

1. Explain the concept.
2. Describe the implementation plan.
3. Show the code for that milestone only.
4. Explain the code.
5. Tell me exactly where to save each file.
6. Give me the terminal commands to run.
7. Explain what successful output should look like.
8. Help me troubleshoot errors if they occur.
9. Give me a small exercise to check my understanding.
10. Wait for me to confirm that I understand and the milestone works before moving to the next major milestone.

Do not skip ahead or generate the entire application at once.

### Rule 4: Explain the machine learning pipeline

Teach me the complete pipeline using this sequence:

Problem definition → Data collection → Data exploration → Data cleaning → Feature engineering → Train/test splitting → Baseline model → Model training → Model evaluation → Model improvement → Probability calibration → Prediction → Deployment → Monitoring.

Explain what each stage means, why it is necessary, and what can go wrong if it is done incorrectly.

### Rule 5: Do not hide complexity behind libraries

Libraries are encouraged, but I must understand what they do.

For example, when using a scikit-learn classifier, explain the underlying idea before using its fit(), predict(), and predict_proba() methods.

Teach me the intuition behind Logistic Regression, Random Forests, feature scaling, classification, overfitting, class imbalance, cross-validation, and probability calibration.

Do not make me implement every algorithm from scratch. Prioritize understanding how to use and evaluate algorithms correctly.

### Rule 6: Do not fabricate results

Never invent historical matches, upcoming fixtures, model accuracy, feature importance, or prediction probabilities.

Run the code and use actual results from the dataset. If data retrieval is unavailable, explain the limitation and provide a documented fallback.

## 5. Phase One: Project Setup

Start by helping me inspect my development environment.

Guide me through:

- Checking the installed Python version.
- Checking whether pip is available.
- Creating a project directory.
- Creating and activating a virtual environment.
- Installing dependencies.
- Creating a requirements.txt file.
- Setting up a clean project structure.
- Initializing Git if appropriate.

Use Windows PowerShell commands by default, but explain equivalent commands for Linux or macOS when useful.

Propose a structure similar to:

```
ucl-prediction-system/
    data/
        raw/
        processed/
    notebooks/
    models/
    src/
        data/
        features/
        training/
        prediction/
    app/
    tests/
    requirements.txt
    README.md
    .gitignore
```

Explain the purpose of every folder. Adjust the structure if a simpler design is more appropriate for the current stage.

Do not create unnecessary empty files.

## 6. Phase Two: Football Data Collection

Help me find reliable, legally accessible sources of historical Champions League match data and current-season fixtures.

Investigate sources such as:

- Football-Data datasets, where suitable historical European competition data is available.
- Open football datasets on GitHub.
- Kaggle datasets, after checking their provenance, coverage, and licensing.
- Official UEFA fixture information or a suitable football data API for current fixtures.

Check the actual coverage and availability of each proposed source before depending on it.

Explain the difference between historical results, historical match statistics, team information, and upcoming fixtures.

Design a data collection module that can retrieve or import the data we need.

At a minimum, historical records should contain:

- Match date.
- Home team.
- Away team.
- Home goals.
- Away goals.
- Competition and season.
- Match stage, when available.

Where reliable data is available, also consider:

- Home and away shots.
- Shots on target.
- Possession.
- Expected goals (xG).
- Red cards.
- Team ratings or Elo ratings.
- Recent team performance.

Do not assume all these features are available in every dataset.

Use a CSV import fallback if an API is unavailable. Keep API keys in environment variables rather than hardcoding them.

Distinguish Champions League data from domestic league data. If we later incorporate domestic league performance, explain how to do so without accidentally leaking future information into training.

Explain data licensing and source attribution.

## 7. Phase Three: Exploratory Data Analysis

Teach me how to explore the historical dataset using pandas.

We should investigate:

- The number of matches available.
- Seasons and competition stages covered.
- Missing values.
- Duplicate records.
- Incorrect dates and team names.
- The distribution of home goals and away goals.
- The proportions of home wins, draws, and away wins.
- Whether home advantage appears in the dataset.
- How the data changes across seasons.

Create useful visualizations and explain what each one tells us.

Explain the difference between an observation, a feature, and a target variable.

Do not remove records automatically without explaining why.

## 8. Phase Four: Define the Prediction Target

For the initial model, predict the three-way match result.

Create a target variable called result:

- H = home win.
- D = draw.
- A = away win.

Generate this target from the historical full-time home and away scores.

Explain why the target must not be included among the model's input features.

Also explain why using final match scores, final possession, or final shots to predict the result before kickoff would cause data leakage.

Keep the first version focused on predicting the result, not the exact score.

Exact-score prediction may be considered as an optional advanced extension after the basic system works.

## 9. Phase Five: Feature Engineering

This is one of the most important learning stages.

Teach me how to create useful pre-match features using only information that would have been available before kickoff.

Start with features such as:

- Home team identity.
- Away team identity.
- Home and away team recent form.
- Average goals scored in recent matches.
- Average goals conceded in recent matches.
- Historical team strength ratings.
- Home advantage.
- Rest days since the previous match.
- Head-to-head statistics, only where justified and carefully calculated.

Use a configurable recent-form window, such as the previous five matches.

For every rolling feature:

- Sort matches chronologically.
- Calculate the feature using earlier matches only.
- Shift rolling statistics appropriately so that the current match cannot influence its own features.
- Handle teams with little or no historical data.
- Prevent future matches from influencing past predictions.

Explain every feature mathematically or with a small football example.

Do not assume that raw team names or head-to-head history will necessarily improve the model. Test whether features add value.

If we use domestic league statistics, ensure those statistics were available at the date of the Champions League match being predicted.

## 10. Phase Six: Training and Testing

Teach me how to divide the dataset into training, validation, and test sets.

Because this is a football prediction task, use chronological evaluation rather than randomly mixing past and future matches.

Train on older seasons or earlier dates and evaluate on later matches.

Explain:

- Training data.
- Validation data.
- Test data.
- Overfitting.
- Generalization.
- Data leakage.
- Why random splitting can create misleading results in this project.

Keep matches from the same date or round from leaking into an earlier training period when that would violate the intended evaluation design.

Use an appropriate baseline, such as predicting the most common outcome or using historical outcome frequencies.

Do not claim that a complex model is better until it has been compared with the baseline on the same held-out matches.

## 11. Phase Seven: Train Multiple Models

Begin with these models:

1. DummyClassifier as a simple baseline.
2. Logistic Regression.
3. Random Forest.
4. A suitable gradient boosting classifier.

Explain the intuition behind each model before implementing it.

Use scikit-learn Pipelines and ColumnTransformer where appropriate so preprocessing is learned from training data and consistently applied to future predictions.

Handle categorical features properly.

If team identity is encoded, explain the encoding strategy and what happens when an unfamiliar team appears.

Do not automatically apply SMOTE or other resampling techniques. First evaluate class distributions and explain whether resampling is justified.

Compare models using the same chronological evaluation framework.

Save the selected trained pipeline and any required metadata.

## 12. Phase Eight: Model Evaluation

Evaluate the models with metrics suitable for three-class football prediction.

Include:

- Accuracy.
- Precision, recall, and F1-score.
- Macro-F1.
- Confusion matrix.
- Multiclass log loss, when valid probabilities are available.
- Multiclass Brier score or another suitable probability-quality metric.

Explain what each metric means using football examples.

Discuss why accuracy alone can be misleading.

Where feasible, evaluate probability calibration and use a suitable chronological validation procedure for calibration.

Compare the model against a simple baseline.

Display the results in a readable table and visualizations.

Do not fabricate scores or claim that a model performs well without evidence.

If the dataset is small, incomplete, or from a different era of football, explain how that affects confidence in the results.

## 13. Phase Nine: Upcoming Fixture Predictions

Build a prediction module that accepts an upcoming fixture and produces:

- Home team.
- Away team.
- Scheduled kickoff, when available.
- Home win probability.
- Draw probability.
- Away win probability.
- Most likely outcome.
- Model version and training data cutoff date.
- A short explanation of the most relevant available factors.

The three probabilities must sum to approximately 100%.

Use predict_proba() only when the selected model supports it and the outputs are valid. Explain that these values are estimates and may require calibration.

Never use the final result or statistics from a match that has not happened yet.

Use the latest information available before the actual prediction timestamp. Store or record prediction timestamps and training data cutoffs.

If a current fixture feed is unavailable, allow the user to select home and away teams manually.

If a team is missing from the training data, handle it safely and clearly explain the limitation instead of silently producing an unreliable prediction.

Do not pretend that manually selected fixtures have been verified against an official fixture source.

## 14. Phase Ten: Streamlit Web Application

Once the data pipeline and model work, build a simple Streamlit interface.

The application should include:

### Dashboard
- Project description.
- Model overview.
- Last training date.
- Historical evaluation metrics.
- Data source information.

### Fixture Predictor
- Select the home team.
- Select the away team.
- Choose an upcoming fixture or enter one manually.
- Display the predicted outcome and the three probabilities.
- Display a clear explanation of uncertainty.

### Upcoming Fixtures
- List upcoming fixtures when reliable fixture data is available.
- Allow users to request predictions for individual fixtures.
- Show kickoff times with the correct timezone.
- Clearly distinguish verified fixtures from manually entered ones.

### Model Insights
- Show the evaluation results.
- Show a confusion matrix.
- Display appropriate feature importance or model interpretation.
- Explain the limitations of feature importance and any model-specific interpretation.

Use a clean, responsive, football-themed design without sacrificing clarity.

Do not add user accounts, betting odds integrations, payment systems, or unnecessary features in the initial version.

## 15. Phase Eleven: Testing and Reliability

Teach me how to test the application.

Include tests for:

- Data loading and validation.
- Target generation.
- Feature engineering.
- Chronological data splitting.
- Prevention of target leakage.
- Prediction probability validity.
- Unknown teams and missing features.
- Saving and loading the model.
- Basic application functionality.

Add validation checks that prevent accidental training on future data.

Explain common errors and how to debug them.

## 16. Phase Twelve: Documentation

Help me write a README that documents:

- Project purpose.
- Features.
- Technology stack.
- Data sources and licenses.
- Installation steps.
- How to train the model.
- How to launch the application.
- Model evaluation results.
- Known limitations.
- Future improvements.

Help me understand the project well enough to explain the complete pipeline in a university project presentation.

## 17. Important ML Principles

Follow these principles throughout development:

1. Historical data must be real and traceable to a source.
2. All prediction features must be available before the match begins.
3. Evaluation must respect time order.
4. Preprocessing must not learn from the test set.
5. Model selection must be based on validation data, not repeatedly on the final test set.
6. Predictions must not be presented as certainties.
7. Probabilities must be evaluated, not assumed to be reliable just because they sum to 100%.
8. The model should be compared with a simple baseline.
9. All important design decisions must be explained.
10. Code should be modular, readable, tested, and reproducible.
11. Record the training data cutoff and the date of each prediction.
12. Clearly distinguish verified data from assumptions and missing information.

## 18. How We Should Begin

Start with **Milestone 1: Understanding the project and preparing my development environment**.

First, give me a simple explanation of how a football match prediction model learns from historical results.

Then explain the planned architecture and the machine learning pipeline.

Next, help me inspect my Python environment and set up the project directory, virtual environment, and dependencies.

Do not implement the complete system in the first response.

Teach me in small steps, explain the code, and wait for my confirmation at each major milestone.

My goal is not merely to have a working application. My goal is to understand Python, machine learning, data engineering, model evaluation, and the complete process of turning historical football data into useful pre-match predictions.