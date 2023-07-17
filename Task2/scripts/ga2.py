import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from deap import algorithms, base, creator, tools
from sklearn.feature_selection import RFE



# Function to ask for the CSV file path
def ask_csv_path():
    file_path = "/Users/nutro/Dropbox/msc_resources/second semester/intelligent systems CSCD612/task2/dataset/data/splice.csv"
    return file_path


# Load the dataset from the CSV file
filename = ask_csv_path()
dataset = pd.read_csv(filename)

# Preprocess string columns and handle missing values
for column in dataset.columns:
    if dataset[column].dtype == 'object':
        dataset[column] = dataset[column].astype('category').cat.codes
    dataset[column].fillna(dataset[column].mean(), inplace=True)

# Extract the features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the evaluation function for the individual
def evaluate_classification(individual):
    # Select features based on the individual's binary string
    selected_features = [feature for feature, select in zip(range(X_train.shape[1]), individual) if select]
    if not selected_features:
        return 0.0, 0.0  # Avoid selecting no features

    # Create a reduced dataset with the selected features
    X_train_reduced = X_train[:, selected_features]

    # Train a classifier (e.g., k-Nearest Neighbors)
    clf = KNeighborsClassifier()  # Set the desired k value
    clf.fit(X_train_reduced, y_train)

    # Calculate accuracy on the test set
    accuracy = clf.score(X_test[:, selected_features], y_test)

    # Calculate feature reduction rate
    reduction_rate = (1 - X_train_reduced.shape[1] / X_train.shape[1]) * 100

    return accuracy, reduction_rate

def evaluate_regression(individual):
    # Select features based on the individual's binary string
    selected_features = [feature for feature, select in zip(range(X_train.shape[1]), individual) if select]
    if not selected_features:
        return float('-inf'), 0.0  # Avoid selecting no features

    # Create a reduced dataset with the selected features
    X_train_reduced = X_train[:, selected_features]

    # Train a regressor (e.g., k-Nearest Neighbors)
    regressor = KNeighborsRegressor()  # Set the desired k value
    regressor.fit(X_train_reduced, y_train)

    # Calculate R-squared score on the test set
    r_squared = regressor.score(X_test[:, selected_features], y_test)

    # Calculate feature reduction rate
    reduction_rate = (1 - X_train_reduced.shape[1] / X_train.shape[1]) * 100

    return r_squared, reduction_rate


# Define the genetic algorithm parameters
POPULATION_SIZE = 40
GENERATIONS = 80
CROSSOVER_PROBABILITY = 0.7
MUTATION_PROBABILITY = 0.35
ELITE_KIDS = int(0.1 * POPULATION_SIZE)
STALLING_GENERATIONS = 10

# Calculate the chromosome length based on the number of features
CHROMOSOME_LENGTH = X_train.shape[1]

# Create the fitness and individual classes
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))  # Maximizing accuracy/R-squared, minimizing reduction rate
creator.create("Individual", list, fitness=creator.Fitness)

# Initialize the toolbox
toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=CHROMOSOME_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the fitness functions
def f1_classification(individual):
    accuracy, reduction_rate = evaluate_classification(individual)
    return accuracy, reduction_rate

def f1_regression(individual):
    r_squared, reduction_rate = evaluate_regression(individual)
    return r_squared, reduction_rate

def f2(individual):
    _, reduction_rate = evaluate_classification(individual)
    return reduction_rate,


# Define the elitist selection based on Hamming distance
def hamming_distance(individual1, individual2):
    return sum(bit1 != bit2 for bit1, bit2 in zip(individual1, individual2))

def elitist_selection(population, k):
    pop_size = len(population)
    distances = np.zeros((pop_size, pop_size))

    # Calculate fitness values for the population
    fitness_values = [toolbox.evaluate(individual) for individual in population]
    for i in range(pop_size):
        population[i].fitness.values = fitness_values[i]

    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            distances[i, j] = distances[j, i] = hamming_distance(population[i], population[j])

    selected = []
    remaining = list(range(pop_size))

    # Check if the number of elite individuals is larger than the population size
    num_elite = min(k, pop_size)

    while len(selected) < num_elite:
        best = min(remaining, key=lambda x: sum(distances[x][s] for s in selected))
        remaining.remove(best)
        selected.append(best)

    return [population[i] for i in selected]


# Define the HUX crossover function
def hux_crossover(ind1, ind2):
    size = len(ind1)
    for i in range(size):
        if random.random() < 0.5:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2


# Define the cataclysmic mutation function
def cataclysmic_mutation(individual):
    for i in range(len(individual)):
        if random.random() < MUTATION_PROBABILITY:
            individual[i] = random.randint(0, 1)

    return individual,


# Register the operators
toolbox.register("select", elitist_selection)  # Elitist selection based on Hamming distance
toolbox.register("mate", hux_crossover)  # HUX crossover operator
toolbox.register("mutate", cataclysmic_mutation)  # Cataclysmic mutation operator

# Determine the fitness function based on the target variable type
if np.issubdtype(y_train.dtype, np.number):
    toolbox.register("evaluate", evaluate_regression)
    toolbox.register("f1", f1_regression)
else:
    toolbox.register("evaluate", evaluate_classification)
    toolbox.register("f1", f1_classification)

# Run the genetic algorithm
population = toolbox.population(n=POPULATION_SIZE)
hof = tools.ParetoFront()  # Hall of Fame to store non-dominated solutions

prev_best_fitness = None  # To track previous best fitness value
stalling_count = 0  # To count generations with no improvement

# Keep track of statistics
gen_fitness = []

for generation in range(GENERATIONS):
    # Select the elite individuals based on Hamming distance
    elites = toolbox.select(population, ELITE_KIDS)

    # Perform HUX crossover on the selected elite individuals
    offspring = [toolbox.clone(ind) for ind in elites]
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Evaluate the offspring population
    for ind in offspring:
        fitness_values = toolbox.evaluate(ind)
        ind.fitness.values = fitness_values

    # Perform cataclysmic mutation if there is convergence
    best_fitness = None
    if len(hof) > 0:
        best_fitness = hof[0].fitness.values

    if prev_best_fitness is not None and best_fitness is not None and np.array_equal(best_fitness, prev_best_fitness):
        stalling_count += 1
        if stalling_count >= STALLING_GENERATIONS:
            for ind in offspring:
                toolbox.mutate(ind)

    # Combine the elite individuals and offspring population
    population = elites + offspring

    # Update the Hall of Fame
    hof.update(population)

    # Store fitness values for the population
    fitness_values = np.array([ind.fitness.values for ind in population])
    gen_fitness.append(fitness_values)

    # Update the statistics
    if len(gen_fitness) > 0:
        record = np.array(gen_fitness)
        best_fitness = np.max(record[:, :, 0], axis=1)
        avg_fitness = np.mean(record[:, :, 0], axis=1)
        std_fitness = np.std(record[:, :, 0], axis=1)
        best_reduction = np.min(record[:, :, 1], axis=1)
        avg_reduction = np.mean(record[:, :, 1], axis=1)
        std_reduction = np.std(record[:, :, 1], axis=1)

        print("Generation:", generation + 1)
        print("Best Fitness:", best_fitness[-1])
        print("Best Reduction Rate:", best_reduction[-1])

        prev_best_fitness = best_fitness[-1]

# Convert gen_fitness to NumPy array
record = np.array(gen_fitness)

# Calculate the best fitness values
best_fitness = np.max(record[:, :, 0], axis=1)

# Retrieve the best individual from the Hall of Fame
if len(hof) > 0:
    best_individual = hof[0]
    selected_features = [feature for feature, select in zip(range(X_train.shape[1]), best_individual) if select]
    # Calculate the reduction rate
    reduction_rate = (1 - len(selected_features) / X_train.shape[1]) * 100
    # Print the fitness values of the best individual
    print("Best Fitness:", best_fitness[0])
    print("Best Reduction Rate:", reduction_rate)
    print("Selected Features:", selected_features)
    print("No. of Selected Features:", len(selected_features))
else:
    print("No solutions found in the Hall of Fame.")
