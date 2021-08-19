from deap import base, creator, tools

import random
import numpy as np
import statsmodels.api as sm
import pandas as pd

from tqdm import tqdm

class Patient_opt:


    def __init__(self, patients, mutpb=0.05, copb=0.5, n_indviduals=100, n_gens=100):
        super().__init__()

        self.patients = patients
        self.mutpb = mutpb
        self.copb = copb
        self.n_indviduals = n_indviduals
        self.n_gens = n_gens

        self.toolbox = base.Toolbox()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,-1.0))
        creator.create("solution", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Initialise each attribute of the member as either 0 or 1 - The grouping function
        self.toolbox.register('grouping', np.random.randint, 0, 2)

        # Use the grouping function to fill each member of the population with 25 different 
        self.toolbox.register('solution', tools.initRepeat, creator.solution, self.toolbox.grouping, n = self.patients.shape[0])

        # Create a population container for a number of potential grouping sets (Solutions)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.solution)

        # Register selection method
        self.toolbox.register("select", tools.selTournament, tournsize=5)

        # Register crossover method
        self.toolbox.register("mate", tools.cxTwoPoint)

        # Register mutate method
        self.toolbox.register("mutate", tools.mutFlipBit, indpb = 0.2)


    def create_population(self):

        # Fill with 100 different member of the population
        self.population = self.toolbox.population(n=self.n_indviduals)


    def evaluate_individual(self, individual):

        duration = self.patients['survival (months)'].values
        death_obs = (self.patients['DiedvsAlive'] == 'Died').values
        groups = individual

        # Perform log-rank test to determine statistic to maximise
        stat, p = sm.duration.survdiff(duration, death_obs, groups)

        # Compute the balance between datasets
        bal = np.abs((len(groups)/2) - np.sum(groups)) 

        # Return cost functions to maximise
        return (stat.astype(np.float16), bal)


    def select_individuals(self):

        new_gen = self.toolbox.select(self.population, len(self.population))

        # Clone the selected individuals
        self.offspring = [self.toolbox.clone(child) for child in new_gen]


    def crossover_individuals(self):

        for child1, child2 in zip(self.offspring[::2], self.offspring[1::2]):

            if random.random() < self.copb:

                self.toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values
    

    def mutate_individuals(self):

        for mutant in self.offspring:
            if random.random() < self.copb:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values


    def check_fitness(self):

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.offspring if not ind.fitness.valid]

        fitnesses = [self.evaluate_individual(indv) for indv in invalid_ind]

        for ind, fit in zip(invalid_ind, fitnesses):

            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        self.population = self.offspring

        self.fitnesses = np.array([fitness[0] for fitness in fitnesses])


    def run_optimisation(self):

        self.create_population()

        self.results = []

        for i in tqdm(range(self.n_gens)):

            self.select_individuals()
            self.crossover_individuals()
            self.mutate_individuals()
            self.check_fitness()

            self.results.append({'Individuals': self.population
                                ,'Fitnesses': self.fitnesses})

