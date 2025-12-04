import os
import pandas as pd
import numpy as np


def load_csv(path: str) -> pd.DataFrame:
	# Attempt to read with no header (the provided CSV appears to be headerless)
	return pd.read_csv(path, header=None, dtype=str)


def is_numeric_series(s: pd.Series) -> pd.Series:
	# Convert to numeric; valid numeric -> notna
	return pd.to_numeric(s, errors="coerce").notna()


def rescale_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] <= 1:
        print("No descriptor columns available for rescaling.")
        return df

    # Exclude the first column (target column)
    descriptor_columns = df.columns[1:]

    for col in descriptor_columns:
        try:
            col_numeric = pd.to_numeric(df[col], errors="coerce")
            col_min = col_numeric.min()
            col_max = col_numeric.max()

            if col_min == col_max:
                # All values are identical; set to 0
                df[col] = 0
                print(f"Column {col} has identical values; rescaled to 0.")
            else:
                # Rescale to [0, 1]
                df[col] = (col_numeric - col_min) / (col_max - col_min)
        except Exception as e:
            print(f"Rescaling failed")

    print("Rescaling from range 0-1 -> DONE.")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    n_rows = len(df)
    if n_rows == 0:
        return df

    # Convert all columns to numeric where possible; keep original strings too
    numeric = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))

    # Determine junk and zero 
    junk_mask = numeric.isna()
    zero_mask = numeric.eq(0)

    # Percentages for filtering junk / zeros
    col_junk_pct = junk_mask.sum(axis=0) / n_rows
    col_zero_pct = zero_mask.sum(axis=0) / n_rows
    col_combined_pct = (junk_mask | zero_mask).sum(axis=0) / n_rows

    # Clean junk and zeros
    drop_cols = set()
    drop_cols.update(col_junk_pct[col_junk_pct > 0.5].index.tolist())
    drop_cols.update(col_zero_pct[col_zero_pct > 0.9].index.tolist())
    drop_cols.update(col_combined_pct[col_combined_pct > 0.7].index.tolist())

    if drop_cols:
        print(f"Dropping {len(drop_cols)} columns by threshold: {sorted(drop_cols)}")
        df.drop(columns=list(drop_cols), inplace=True)
        numeric.drop(columns=list(drop_cols), inplace=True)
        junk_mask.drop(columns=list(drop_cols), inplace=True)
        zero_mask.drop(columns=list(drop_cols), inplace=True)
    else:
        print("No columns dropped.")

    # recompute df after done with editing columns
    n_cols_after = df.shape[1]
    if n_cols_after == 0:
        print("No columns remain after dropping. Returning empty dataframe.")
        return df

    # Row rule: drop any row with >70% combined junk or zero values (on remaining columns)
    combined_mask = junk_mask | zero_mask
    row_combined_pct = combined_mask.sum(axis=1) / n_cols_after
    rows_to_drop = row_combined_pct[row_combined_pct > 0.7].index.tolist()
    if rows_to_drop:
        print(f"Dropping {len(rows_to_drop)} rows (>70% combined junk/zero rule)")
        df.drop(index=rows_to_drop, inplace=True)
        # we don't need to update numeric/junk masks further for saved output
    else:
        print("No rows dropped.")

    # Column K replacement of junk values with column mean -> column index 10 
    column_k = None
    if df.shape[1] > 10:  # Ensure there are at least 11 columns remaining
        column_k = df.columns[10]

    if column_k is None:
        print("Column K not found (fewer than 11 columns) - skipping junk replacement.")
        return df

    # Compute numeric series for column K from the current df (after drops/rows removed)
    col_numeric = pd.to_numeric(df[column_k], errors="coerce")
    valid_mean = col_numeric.dropna().mean()
    if pd.isna(valid_mean):
        print(f"No valid numeric values found in column K; skipping replacement.")
        return df

    # Replace junk (non-numeric) values in column K with mean
    junk_positions = col_numeric.isna()
    n_junk = junk_positions.sum()
    if n_junk:
        print(f"Replacing {n_junk} junk values in column K with mean={valid_mean}")
        # assign as numeric value (float)
        df.loc[junk_positions, column_k] = valid_mean

    return df


def sort_by_pIC50(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.sort_values(by=df.columns[0], ascending=False)
        print("Sorted dataset by pIC50 (first column) in descending order.")
    except Exception as e:
        print(f"Error sorting by pIC50: {e}")
    return df


def split_dataset(df: pd.DataFrame, train_path: str, validate_path: str, test_path: str):
    try:
        train, validate, test = [], [], []

        for i, row in df.iterrows():
            if i % 4 in [0, 1]:
                train.append(row) # 2 out of 4 to train
            elif i % 4 == 2:
                validate.append(row) # 1 out of 4 to validate
            else:
                test.append(row) # 1 out of 4 to test

        train_df = pd.DataFrame(train, columns=df.columns)
        validate_df = pd.DataFrame(validate, columns=df.columns)
        test_df = pd.DataFrame(test, columns=df.columns)

        train_df.to_csv(train_path, index=False, header=False)
        validate_df.to_csv(validate_path, index=False, header=False)
        test_df.to_csv(test_path, index=False, header=False)

        print(f"Train.csv written with {len(train_df)} rows.")
        print(f"Validate.csv written with {len(validate_df)} rows.")
        print(f"Test.csv written with {len(test_df)} rows.")
    except Exception as e:
        print(f"Error splitting dataset: {e}")


def split_target_descriptors(input_path: str, x_output_path: str, y_output_path: str):
    try:
        df = pd.read_csv(input_path, header=None)
        y = df.iloc[:, 0]  # First column as target
        x = df.iloc[:, 1:]  # Remaining columns as descriptors

        y.to_csv(y_output_path, index=False, header=False)
        x.to_csv(x_output_path, index=False, header=False)

        print(f"Split {input_path} into {x_output_path} (descriptors) and {y_output_path} (target).")
    except Exception as e:
        print(f"Error splitting target and descriptors for {input_path}: {e}")


def main(in_path: str):
    if not os.path.exists(in_path):
        print(f"Input file not found: {in_path}")
        return

    print(f"Loading {in_path} ...")
    df = load_csv(in_path)
    print(f"Input shape: {df.shape}")

    # Convert IC50 (column 0) to pIC50
    # IC50 values are assumed to be in nM (nanomolar)
    print("Converting IC50 (nM) to pIC50...")
    ic50_values = pd.to_numeric(df[0], errors="coerce")
    # Convert nM to M: IC50_M = IC50_nM * 10^-9
    # pIC50 = -log10(IC50_M) = -log10(IC50_nM * 10^-9) = -log10(IC50_nM) - log10(10^-9) = -log10(IC50_nM) + 9
    df[0] = 9 - np.log10(ic50_values)
    print("IC50 to pIC50 conversion complete.")

    # Clean, rescale, and sort the data
    cleaned = clean_dataframe(df)
    cleaned_scaled = rescale_descriptors(cleaned)
    
    # Sort by pIC50 (first column) in descending order
    cleaned_scaled_sorted = sort_by_pIC50(cleaned_scaled)
    
    # Reset index after sorting to ensure proper sequential splitting
    cleaned_scaled_sorted = cleaned_scaled_sorted.reset_index(drop=True)

    # Split the dataset into Train, Validate, and Test files (temporary)
    split_dataset(cleaned_scaled_sorted, "Train.csv", "Validate.csv", "Test.csv")

    # Split Train, Validate, and Test into target and descriptors with final output names
    split_target_descriptors("Train.csv", "Train-Data.csv", "Train-pIC50.csv")
    split_target_descriptors("Validate.csv", "Validation-Data.csv", "Validation-pIC50.csv")
    split_target_descriptors("Test.csv", "Test-Data.csv", "Test-pIC50.csv")

    # Remove temporary files
    try:
        os.remove("Train.csv")
        os.remove("Validate.csv")
        os.remove("Test.csv")
        print("\nTemporary files removed.")
    except Exception as e:
        print(f"Error removing temporary files: {e}")

    print("\n=== Cleaning Complete ===")
    print("Generated 6 files:")
    print("  - Train-Data.csv")
    print("  - Train-pIC50.csv")
    print("  - Validation-Data.csv")
    print("  - Validation-pIC50.csv")
    print("  - Test-Data.csv")
    print("  - Test-pIC50.csv")


if __name__ == "__main__":
    main("Alzheimer2.csv")


# ============================================================================
# ORIGINAL CODE (POST-CLEANING) - COMMENTED OUT FOR LATER USE
# ============================================================================

# import time  # provides timing for benchmarks
# from numpy import *  # provides complex math and array functions
# from sklearn import svm  # provides Support Vector Regression
# from sklearn import linear_model
# from sklearn import neural_network
# from sklearn import cross_decomposition  # provides Partial Least Square Regression
# import csv
# import math
# import sys

# # Local files created by me
# import mlr
# import FromDataFileMLR
# import FromFinessFileMLR


# class FitnessAnalyzer:
#     def __init__(self):
#         self.fitnessdata = FromFinessFileMLR.FitnessResults()

#     # ------------------------------------------------------------------------------
#     def getAValidrow(self, numOfFea, eps=0.015):

#     # In the parameters, numOfFea refers to the number of columns in the data.
#     # This number can approximately be 751 columns that starts from column 2 until
#     # column 750. Some of you may get more and some may get less features (column)
#     # depending on how your clean up program is written. This routine returns a
#     # vector of 1 row with 750 column. The contents of the columns are either
#     # 0 or 1 as shown in the following pseudo code:

#     #  while  (true)
#     #  {
#     #     V = a vector with 1 row and 750 columns where each bit is initialized to 0
#     #     for (i=0, i<numOfFea; i++)
#     #     {
#     #        int r = random.uniform(0, 1) # this returns a value between 0 and 1
#     #        if (r <= eps)
#     #          V[i] = 1;
#     #         else
#     #           V[i] = 0;
#     #      }
#     #      count = count the number of 1's in the vector
#     #      if (count >=5 and count <=15)
#     #         retun V;
#     #  }

#     # ------------------------------------------------------------------------------
#     def Create_A_Population(self, numOfPop, numOfFea):
#         # The numOfPop that comes to this routine is 50 (indexed from 0 to 49 index)
#         # this routine creates the first poprlation. We want to have 50 rows
#         # (50 population) and 750 columns where columns refer to feature.
#         # Since this is the first population, all 50 rows should be randomly selected.
#         # Therefore, for each row, you need to call the getAValidrow function
#         # and once the vector is returned, it is placed in the position. The peseduo code is:

#         # initialize the population first
#         # for (i = 0; i <numOfPop; i++)
#         #     for  (j =0; j< numOfFea)
#         #         population [i, j] = 0;
#         #
#         # for (i = 0; i <numOfPop; i++)
#         # {
#         #    V =  getAValidrow(self, numOfFea, 0.015):
#         #    population [i] = V;
#         # }
#         return population

#     # ------------------------------------------------------------------------------
#     # The following creates an output file. Every time a model is created the
#     # descriptors of the model, the name of the model (ex: "MLR" for multiple
#     # linear regression of "SVM" support vector machine), the R^2 of training, Q^2
#     # of training,R^2 of validation, and R^2 of test is placed in the output file

#     def createAnOutputFile(self):
#         file_name = None
#         algorithm = None

#         timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
#         if ((file_name == None) and (algorithm != None)):
#             file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__,
#                                                     algorithm.model.__class__.__name__, algorithm.gen_max, timestamp)
#         elif file_name == None:
#             file_name = "{}.csv".format(timestamp)
#         fileOut = open(file_name, 'w')
#         fileW = csv.writer(fileOut)

#         fileW.writerow(['Descriptor ID', 'Fitness', 'Model', 'R2', 'Q2', 'R2Pred_Validation', 'R2Pred_Test'])

#         return fileW

#     # -------------------------------------------------------------------------------------------
#     def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
#         #   NewPopulation = create a 2-dimensional array of (numOfPop by num of features)
#         #   Move the 2 best rows of the old population (the two with the lowest fitness values)
#         #   to rows 1 and 2 of the NewPopulation. Do the following:
#         #   Mom = NewPopulation[0];
#         #   Dad = NewPopulation[1];
#         #   OnePointCrossOver = select a random number between 0 up to numOfFea. The
#         #   formula in python is:   x = random.randint(0, numOfFea)
#         #
#         #   Child1[0:x] = Mom[0:x]
#         #   Child1[x+1:numOfFea] = Dad[x+1, numOfFea]
#         #   Child2[0:x] = Dad[0:x]
#         #   Child1[x+1:numOfFea] = Mom[x+1, numOfFea]
#         #   NewPopulation[2] = Child1;
#         #   NewPopulation[3] = Child2;
#         #
#         #   Now in the new population, we have 4 rows entere. Starting from the 5th row until
#         #   the last row (row 50), you should do as you did for creating the first population.

#         return NewPopulation

#     # -------------------------------------------------------------------------------------------
#     def PerformOneMillionIteration(self, numOdPop, numOfFea, population, fitness, model, fileW,
#                                    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
#         NumOfGenerations = 1
#         OldPopulation = population
#         while (NumOfGenerations < 1000):
#             population = self.createANewPopulation(numOdPop, numOfFea, OldPopulation, fitness)
#             fittingStatus, fitness = self.fitnessdata.validate_model(model, fileW, population,
#                                                                      TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
#             NumOfGenerations = NumOfGenerations + 1
#             print(NumOfGenerations)
#         return
#     # --------------------------------------------------------------------------------------------


# def main():
#     # create an object of Multiple Linear Regression model.
#     # The class is located in mlr file
#     model = mlr.MLR()
#     # model = svm.LinearSVR()
#     # model = cross_decomposition.PLSRegression()

#     filedata = FromDataFileMLR.DataFromFile()  # constructor for FromDataFileMLR class
#     fitnessdata = FromFinessFileMLR.FitnessResults()  # constructor for FromFitnessFileMLR object
#     analyzer = FitnessAnalyzer()  # constructor for MainMLR class

#     # create an output file. Name the object to be FileW
#     fileW = analyzer.createAnOutputFile()

#     # Number of descriptor should be 385 and number of population should be 50 or more
#     numOfPop = 50
#     numOfFea = 385

#     # we continue exhancing the model; however if after 1000 iteration no
#     # enhancement is done, we can quit
#     unfit = 1000

#     # Final model requirements: The following is used to evaluate each model. The minimum
#     # values for R^2 of training should be 0.6, R^2 of Validation should be 0.5 and R^2 of
#     # test should be 0.5
#     R2req_train = .5
#     R2req_validate = .5
#     R2req_test = .5

#     # getAllOfTheData is in FromDataFileMLR file. The following places the data
#     # (training data, validation data, and test data) into associated matrices
#     TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = filedata.getAllOfTheData()  # gets descriptor values and target values
#     TrainX, ValidateX, TestX = filedata.rescaleTheData(TrainX, ValidateX, TestX)

#     fittingStatus = unfit
#     population = analyzer.Create_A_Population(numOfPop, numOfFea)
#     fittingStatus, fitness = fitnessdata.validate_model(model, fileW, population,
#                                                         TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

#     analyzer.PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW,
#                                         TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)


# # main routine ends in here
# # ------------------------------------------------------------------------------
# # main()
# # ------------------------------------------------------------------------------


