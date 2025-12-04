import time  # provides timing for benchmarks
from numpy import *  # provides complex math and array functions
#from sklearn import svm  # provides Support Vector Regression
#from sklearn import linear_model
#from sklearn import neural_network
#from sklearn import cross_decomposition  #provides Partial Least Square Regression
import csv
import math
import sys

#Local files created by me
import mlr
import FromDataFileMLR
import FromFinessFileMLR

class FitnessAnalyzer:
    def __init__(self):
        self.fitnessdata = FromFinessFileMLR.FitnessResults()
    # ------------------------------------------------------------------------------
    def getAValidrow(self, numOfFea, eps=0.015):
        # In the parameters, numOfFea refers to the number of columns in the data. 
        # This number is 385. In this routune, you need to select a random number 
        # between 0 and 1: {r = random.uniform(0, 1)}. If the value is less that 
        # or equal 0.015 (1.5% of the features) you can set that index to 1; otherwise,
        # the index is set to zero.
        # After going through the entire 385 features, you can add all the numbers 
        # together to see if at least 3 features are set to 1. If that is the case
        # you can return the vector; otherwise, you need to do it again.

        valid = False
        while not valid:
            V = zeros(numOfFea, dtype=int)
            for j in range(numOfFea):
                r = random.uniform(0,1)
                if r <= eps:
                    V[j] = 1
            
            if V.sum() >= 3:
                valid = True

        return V
    # ------------------------------------------------------------------------------
    def Create_A_Population(self, numOfPop, numOfFea):
        # this routine creates the first poprlation. We want to have 50 rows
        # (50 population) and number of features is 385 (numOfFea = 385). 
        # Since this the first population, all 50 rows should be randomly selected. 
        # Therefore, for each row, you need to call the getAValidrow function 
        # and move the vector into the row. Once the first population is created
        # you can return it to Callee

        population = zeros((numOfPop, numOfFea), dtype=int)

        for i in range(numOfPop):
            population[i, :] = self.getAValidrow(numOfFea)
 
        return population
    # ------------------------------------------------------------------------------
    # The following creates an output file. Every time a model is created the
    # descriptors of the model, the name of the model (ex: "MLR" for multiple
    # linear regression of "SVM" support vector machine), the R^2 of training, Q^2
    # of training,R^2 of validation, and R^2 of test is placed in the output file

    def createAnOutputFile(self):
        file_name = None
        algorithm = None

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if ((file_name == None) and (algorithm != None)):
            file_name = "{}_{}_gen{}_{}.csv".format(algorithm.__class__.__name__,
                                                    algorithm.model.__class__.__name__, algorithm.gen_max, timestamp)
        elif file_name == None:
            file_name = "{}.csv".format(timestamp)
        fileOut = open(file_name, 'w')
        fileW = csv.writer(fileOut)

        fileW.writerow(['Descriptor ID', 'Fitness', 'Model', 'R2', 'Q2', 'R2Pred_Validation', 'R2Pred_Test'])

        return fileW
    # -------------------------------------------------------------------------------------------
    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
        #   NewPopulation = create a 2D array of (numOfPop by num of features)
        #   sort the OldPopulation and their fitness value based on the asending
        #   order of the fitness. The lower is the fitness, the better it is.
        #   So, Move two rows with of the OldPopulation with the lowest fitness
        #   to row 1 and row 2 of the new population.
        #
        #   Name the first row to be Dad and the second row to be mom. Create a
        #   one point or n point cross over to create at least couple of children.
        #   children should be moved to the third, fourth, fifth, etc rows of the
        #   new population.
        #   The rest of the rows should be filled randomly the same way you did when
        #   you created the initial population.
        NewPopulation = zeros((numOfPop, numOfFea), dtype=int)

        # Sort by fitness (ascending: lower = better)
        sorted_idx = fitness.argsort()

        # Elitism: best two become Dad and Mom in first two rows
        dad = OldPopulation[sorted_idx[0]].copy()
        NewPopulation[0, :] = dad

        if numOfPop > 1:
            mom = OldPopulation[sorted_idx[1]].copy()
            NewPopulation[1, :] = mom
        else:
            # degenerate case; just duplicate dad
            mom = dad

        next_row = 2

        # Create at least one child if there is room
        if numOfPop - next_row >= 1:
            # One-point crossover index in [1, numOfFea-1] when possible
            if numOfFea > 1:
                cp = random.randint(1, numOfFea)  # high is exclusive → 1..numOfFea-1
            else:
                cp = 0

            child1 = dad.copy()
            child1[cp:] = mom[cp:]

            # Ensure child has at least 3 selected features
            if child1.sum() < 3:
                child1 = self.getAValidrow(numOfFea)

            NewPopulation[next_row, :] = child1
            next_row += 1

        # Create a second child if there is still room
        if numOfPop - next_row >= 1:
            if numOfFea > 1:
                cp2 = random.randint(1, numOfFea)
            else:
                cp2 = 0

            child2 = mom.copy()
            child2[cp2:] = dad[cp2:]

            if child2.sum() < 3:
                child2 = self.getAValidrow(numOfFea)

            NewPopulation[next_row, :] = child2
            next_row += 1

        # Fill the rest randomly like the initial population
        while next_row < numOfPop:
            NewPopulation[next_row, :] = self.getAValidrow(numOfFea)
            next_row += 1

        return NewPopulation

        
    # -------------------------------------------------------------------------------------------
    def PerformOneMillionIteration(self, numOdPop, numOfFea, population, fitness, model, fileW,
                                   TrainX, TrainY, ValidateX, ValidateY, TestX, TestY):
        NumOfGenerations = 1
        OldPopulation = population
        while (NumOfGenerations < 1000):
            population = self.createANewPopulation(numOdPop, numOfFea, OldPopulation, fitness)
            fittingStatus, fitness = self.fitnessdata.validate_model(model, fileW, population,
                                            TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)
            NumOfGenerations = NumOfGenerations + 1
            print(NumOfGenerations)
        return
    # --------------------------------------------------------------------------------------------

def main():
    # create an object of Multiple Linear Regression model.
    # The class is located in mlr file
    model = mlr.MLR()
    #model = svm.LinearSVR()
    #model = cross_decomposition.PLSRegression()

    filedata = FromDataFileMLR.DataFromFile()           # constructor for FromDataFileMLR class
    fitnessdata = FromFinessFileMLR.FitnessResults()    # constructor for FromFitnessFileMLR object
    analyzer = FitnessAnalyzer()                        # constructor for MainMLR class

    # create an output file. Name the object to be FileW
    fileW = analyzer.createAnOutputFile()

    # Number of descriptor should be 385 and number of population should be 50 or more
    numOfPop = 50
    numOfFea = 385

    # we continue exhancing the model; however if after 1000 iteration no
    # enhancement is done, we can quit
    unfit = 1000

    # Final model requirements: The following is used to evaluate each model. The minimum
    # values for R^2 of training should be 0.6, R^2 of Validation should be 0.5 and R^2 of
    # test should be 0.5
    R2req_train = .6
    R2req_validate = .5
    R2req_test = .5

    # getAllOfTheData is in FromDataFileMLR file. The following places the data
    # (training data, validation data, and test data) into associated matrices
    TrainX, TrainY, ValidateX, ValidateY, TestX, TestY = filedata.getAllOfTheData() # gets descriptor values and target values
    TrainX, ValidateX, TestX = filedata.rescaleTheData(TrainX, ValidateX, TestX)

    fittingStatus = unfit
    population = analyzer.Create_A_Population(numOfPop, numOfFea)
    fittingStatus, fitness = fitnessdata.validate_model(model, fileW, population,
                                                              TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)

    analyzer.PerformOneMillionIteration(numOfPop, numOfFea, population, fitness, model, fileW,
                               TrainX, TrainY, ValidateX, ValidateY, TestX, TestY)


# main routine ends in here
# ------------------------------------------------------------------------------
main()
# ------------------------------------------------------------------------------





'''
    # ------------------------------------------------------------------------------
  
    # ------------------------------------------------------------------------------
    def createANewPopulation(self, numOfPop, numOfFea, OldPopulation, fitness):
        # Create a new population using:
        #  - elitism: keep the two best individuals (lowest fitness)
        #  - crossover: create at least two children from Dad & Mom
        #  - random fill: rest of rows are new random individuals (like initial pop)

        # Allocate new population
        NewPopulation = zeros((numOfPop, numOfFea), dtype=int)

        # Sort by fitness (ascending: lower = better)
        sorted_idx = fitness.argsort()

        # Elitism: best two become Dad and Mom in first two rows
        dad = OldPopulation[sorted_idx[0]].copy()
        NewPopulation[0, :] = dad

        if numOfPop > 1:
            mom = OldPopulation[sorted_idx[1]].copy()
            NewPopulation[1, :] = mom
        else:
            # degenerate case; just duplicate dad
            mom = dad

        next_row = 2

        # Create at least one child if there is room
        if numOfPop - next_row >= 1:
            # One-point crossover index in [1, numOfFea-1] when possible
            if numOfFea > 1:
                cp = random.randint(1, numOfFea)  # high is exclusive → 1..numOfFea-1
            else:
                cp = 0

            child1 = dad.copy()
            child1[cp:] = mom[cp:]

            # Ensure child has at least 3 selected features
            if child1.sum() < 3:
                child1 = self.getAValidrow(numOfFea)

            NewPopulation[next_row, :] = child1
            next_row += 1

        # Create a second child if there is still room
        if numOfPop - next_row >= 1:
            if numOfFea > 1:
                cp2 = random.randint(1, numOfFea)
            else:
                cp2 = 0

            child2 = mom.copy()
            child2[cp2:] = dad[cp2:]

            if child2.sum() < 3:
                child2 = self.getAValidrow(numOfFea)

            NewPopulation[next_row, :] = child2
            next_row += 1

        # Fill the rest randomly like the initial population
        while next_row < numOfPop:
            NewPopulation[next_row, :] = self.getAValidrow(numOfFea)
            next_row += 1

        return NewPopulation
'''