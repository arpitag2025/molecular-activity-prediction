# Molecular Activity Prediction with Genetic Feature Selection

A machine learning and data mining project for predicting molecular **pIC50 values** from chemical descriptors. The project combines **Multiple Linear Regression (MLR)** with a **genetic-algorithm-inspired feature selection approach** to identify subsets of molecular descriptors that provide strong predictive performance.

## Overview

Molecular datasets can contain hundreds of descriptors representing different chemical properties. Using every available descriptor can increase model complexity and introduce features that provide little predictive value.

This project searches for smaller, more useful subsets of molecular descriptors and evaluates how effectively they can predict **pIC50**, a measure related to the biological potency of a compound.

The pipeline:

1. Cleans and preprocesses molecular descriptor data.
2. Converts IC50 measurements to pIC50.
3. Splits the data into training, validation, and test sets.
4. Standardizes molecular descriptors.
5. Generates candidate subsets from 385 available descriptors.
6. Trains Multiple Linear Regression models using selected descriptors.
7. Evaluates each model using cross-validation and predictive performance metrics.
8. Uses a genetic-algorithm-inspired search process to explore new feature combinations.

## Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* Multiple Linear Regression
* Feature Selection
* Cross-Validation
* Data Preprocessing
* Genetic Algorithm Concepts

## Project Structure

### `MainMLR.py`

Controls the overall feature-selection and model-evaluation process.

The program begins by generating a population of candidate feature subsets. Each candidate is represented by a binary vector indicating which of the 385 molecular descriptors should be included in the model.

For example:

```text
[0, 1, 0, 0, 1, 0, 1, ...]
```

A value of `1` indicates that the corresponding descriptor is selected, while `0` indicates that it is excluded.

Candidate subsets are evaluated, and higher-performing candidates are used to generate new populations through selection and crossover.

### `mlr.py`

Implements the Multiple Linear Regression model using NumPy.

The model:

* Adds an intercept term to the input data.
* Estimates regression coefficients using least-squares optimization.
* Generates predictions for unseen observations.

### `FromDataFileMLR.py`

Handles loading and preprocessing of the training, validation, and test datasets.

It also standardizes molecular descriptors based on statistics calculated from the training dataset.

### `FromFinessFileMLR.py`

Evaluates candidate models and feature subsets.

For each candidate descriptor subset, the program:

* Selects the corresponding features.
* Fits an MLR model.
* Performs leave-one-out cross-validation.
* Generates validation and test predictions.
* Calculates model-performance metrics.
* Calculates a fitness score.
* Records model results.

### `cleaning.py`

Contains the data preprocessing pipeline, including:

* IC50 to pIC50 conversion
* Detection of non-numeric or invalid values
* Removal of features containing excessive missing or zero values
* Missing-value handling
* Descriptor rescaling
* Dataset sorting
* Training/validation/test splitting
* Separation of target values from molecular descriptors

## Feature Selection

The project uses a genetic-algorithm-inspired approach to search through possible combinations of molecular descriptors.

### Initial Population

The algorithm creates a population of **50 candidate models**.

Each candidate randomly selects a small subset of the **385 available descriptors**.

### Fitness Evaluation

Each candidate descriptor subset is used to train an MLR model. Its predictive performance is then evaluated and converted into a fitness score.

Lower fitness values represent better-performing candidate feature sets.

### Selection and Crossover

The two candidates with the best fitness scores are retained as parent solutions.

A one-point crossover operation combines their descriptor selections to generate new candidate feature sets.

Additional randomly generated candidates are added to maintain population diversity.

### Iterative Search

The process is repeated across generations, allowing the algorithm to search different combinations of descriptors for subsets that produce stronger predictive models.

## Model Evaluation

Models are evaluated using multiple metrics rather than training performance alone:

* **R²** — measures model fit on the training data.
* **Q² / Leave-One-Out Cross-Validation** — evaluates how well the model generalizes when individual training observations are excluded.
* **Validation Predictive R²** — measures performance on the validation dataset.
* **Test Predictive R²** — measures performance on previously unseen test data.
* **Fitness Score** — combines prediction error with model complexity to compare descriptor subsets.

Using separate training, validation, and test datasets helps evaluate whether a selected model generalizes beyond the data used to fit it.

## Machine Learning Pipeline

```text
Raw Molecular Data
        ↓
Data Cleaning
        ↓
IC50 → pIC50 Conversion
        ↓
Feature Filtering
        ↓
Train / Validation / Test Split
        ↓
Descriptor Standardization
        ↓
Generate Candidate Feature Subsets
        ↓
Multiple Linear Regression
        ↓
Cross-Validation & Model Evaluation
        ↓
Fitness Calculation
        ↓
Selection + Crossover
        ↓
New Generation of Feature Subsets
        ↓
Repeat Search
```

## Key Concepts Demonstrated

This project demonstrates practical experience with:

* Machine learning
* Regression modeling
* Feature selection
* Genetic algorithm concepts
* Model validation
* Cross-validation
* Predictive modeling
* High-dimensional data
* Data cleaning and preprocessing
* Numerical computing with NumPy
* Data manipulation with Pandas

## Goal

The goal of the project is to identify a compact subset of molecular descriptors capable of predicting pIC50 while maintaining strong performance on validation and test data.

Rather than simply fitting a regression model using every available descriptor, the project explores the feature space to identify combinations that provide useful predictive information while reducing unnecessary model complexity.
