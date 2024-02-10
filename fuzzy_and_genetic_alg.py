import importlib
if importlib.util.find_spec("skfuzzy") is None:
    !pip install scikit-fuzzy
if importlib.util.find_spec("EasyGA") is None:
    !pip install EasyGA

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import EasyGA
import random
import pandas as pd
import os


# prevents database lock
if os.path.exists('database.db'):
    os.remove('database.db')
if os.path.exists('database.db-journal'):
    os.remove('database.db-journal')

file_name = "tipper_train.csv"
train_data = pd.read_csv(file_name)
# clip values below 0 to exactly 0, with values above 1 set to exactly 1
train_data.iloc[:, 0:6] = train_data.iloc[:, 0:6].clip(0, 1)
train_data.iloc[:, 0:6] = train_data.iloc[:, 0:6] * 10

file_name = "tipper_test.csv"
test_data = pd.read_csv(file_name)
# clip values below 0 to exactly 0, with values above 1 set to exactly 1
test_data.iloc[:, 0:6] = test_data.iloc[:, 0:6].clip(0, 1)
test_data.iloc[:, 0:6] = test_data.iloc[:, 0:6] * 10

def setup_fuzzy_system(chromosome):
  # inputs
  temperature = ctrl.Antecedent(np.linspace(0, 10, 11),'temperature')
  flavor = ctrl.Antecedent(np.linspace(0, 10, 11),'flavor')
  portion_size = ctrl.Antecedent(np.linspace(0, 10, 11),'portion_size')
  attentiveness = ctrl.Antecedent(np.linspace(0, 10, 11),'attentiveness')
  friendliness = ctrl.Antecedent(np.linspace(0, 10, 11),'friendliness')
  speed_of_service = ctrl.Antecedent(np.linspace(0, 10, 11),'speed_of_service')
  food_quality_in = ctrl.Antecedent(np.linspace(0, 10, 11), 'food_quality_in')
  service_quality_in = ctrl.Antecedent(np.linspace(0, 10, 11), 'service_quality_in')

  # outputs
  service_quality = ctrl.Consequent(np.linspace(0, 10, 11), 'service_quality')
  food_quality = ctrl.Consequent(np.linspace(0, 10, 11), 'food_quality')
  tip = ctrl.Consequent(np.linspace(0, 30, 31), 'tip')

  all_input_variables = [temperature, flavor, portion_size, attentiveness, friendliness, speed_of_service, food_quality_in, service_quality_in]
  all_output_variables = [service_quality, food_quality, tip]

  array_shape = (3, 3)

  # membership funtions using chromosome
  for j, var in enumerate(all_input_variables + all_output_variables):
    chromosome_matrix = np.array(chromosome.gene_value_list[j]).reshape(array_shape)
    i=0
    param_a = chromosome_matrix[i, 0]
    param_b = chromosome_matrix[i, 1]
    param_c = chromosome_matrix[i, 2]
    var['poor'] = fuzz.trimf(var.universe,  [param_a, param_b, param_c])

    i=1
    param_a = chromosome_matrix[i, 0]
    param_b = chromosome_matrix[i, 1]
    param_c = chromosome_matrix[i, 2]
    var['average'] = fuzz.trimf(var.universe,  [param_a, param_b, param_c])

    i=2
    param_a = chromosome_matrix[i, 0]
    param_b = chromosome_matrix[i, 1]
    param_c = chromosome_matrix[i, 2]
    var['good'] = fuzz.trimf(var.universe,  [param_a, param_b, param_c])

  tip['poor'] = fuzz.trimf(tip.universe, [0, 0, 30])
  tip['average'] = fuzz.trimf(tip.universe, [0, 15, 30])
  tip['good'] = fuzz.trimf(tip.universe, [15, 15, 30])


  # all fuzzy rules
  # if any are poor: poor, if any are avg: avg, if flavor is good: good
  # if flavor is good but others are bad: avg
  food_rules = []
  food_rules.append(ctrl.Rule(temperature['poor'] | flavor['poor'] | portion_size['poor'], food_quality['poor']))
  food_rules.append(ctrl.Rule(temperature['average'] | flavor['average'] | portion_size['average'], food_quality['average']))
  food_rules.append(ctrl.Rule(flavor['good'], food_quality['good']))
  food_rules.append(ctrl.Rule(temperature['poor'] & flavor['good'] & portion_size['poor'], food_quality['average']))
  food_rules.append(ctrl.Rule(temperature['good'] & flavor['good'] & portion_size['poor'], food_quality['average']))
  food_rules.append(ctrl.Rule(temperature['poor'] & flavor['good'] & portion_size['good'], food_quality['average']))

  service_rules = []
  service_rules.append(ctrl.Rule(attentiveness['poor'] | friendliness['poor'] | speed_of_service['poor'], service_quality['poor']))
  service_rules.append(ctrl.Rule(attentiveness['average'] | friendliness['average'] | speed_of_service['average'], service_quality['average']))
  service_rules.append(ctrl.Rule(friendliness['good'], service_quality['good']))
  service_rules.append(ctrl.Rule(attentiveness['poor'] & friendliness['good'] & speed_of_service['poor'], service_quality['average']))
  service_rules.append(ctrl.Rule(attentiveness['good'] & friendliness['good'] & speed_of_service['poor'], service_quality['average']))
  service_rules.append(ctrl.Rule(attentiveness['poor'] & friendliness['good'] & speed_of_service['good'], service_quality['average']))

  tip_rules = []
  tip_rules.append(ctrl.Rule(food_quality_in['poor'] | service_quality_in['poor'], tip['poor']))
  tip_rules.append(ctrl.Rule(food_quality_in['average'] & service_quality_in['average'], tip['average']))
  tip_rules.append(ctrl.Rule(food_quality_in['good'] & service_quality_in['average'], tip['average']))
  tip_rules.append(ctrl.Rule(food_quality_in['average'] & service_quality_in['good'], tip['good']))
  tip_rules.append(ctrl.Rule(food_quality_in['good'] & service_quality_in['good'], tip['good']))

  # simulation controller
  food_ctrl = ctrl.ControlSystem(food_rules)
  service_ctrl = ctrl.ControlSystem(service_rules)
  tip_ctrl = ctrl.ControlSystem(tip_rules)

  return food_ctrl, service_ctrl, tip_ctrl

# Fitness functions
def execute_fuzzy_inference(food_ctrl, service_ctrl, tip_ctrl, inputs):
  food_sim = ctrl.ControlSystemSimulation(food_ctrl)
  # pass user inputs
  food_sim.input['temperature'] = inputs['temperature']
  food_sim.input['flavor'] = inputs['flavor']
  food_sim.input['portion_size'] = inputs['portion_size']
  # fuzzy inference and defuzzification
  food_sim.compute()

  service_sim = ctrl.ControlSystemSimulation(service_ctrl)
  # pass user inputs
  service_sim.input['attentiveness'] = inputs['attentiveness']
  service_sim.input['friendliness'] = inputs['friendliness']
  service_sim.input['speed_of_service'] = inputs['speed_of_service']
  # fuzzy inference and defuzzification
  service_sim.compute()

  tip_sim = ctrl.ControlSystemSimulation(tip_ctrl)
  # pass inputs
  tip_sim.input['food_quality_in'] = food_sim.output['food_quality']
  tip_sim.input['service_quality_in'] = service_sim.output['service_quality']
  # fuzzy inference and defuzzification
  tip_sim.compute()

  # output final value
  return tip_sim.output['tip']


def fitness_train(chromosome):
  total_error = 0
  for index, row in train_data.iterrows():
    inputs = {
      'temperature': float(row['food temperature']),
      'flavor': float(row['food flavor']),
      'portion_size': float(row['portion size']),
      'attentiveness': float(row['attentiveness']),
      'friendliness': float(row['friendliness']),
      'speed_of_service': float(row['speed of service'])
    }
    actual_tip = float(row['tip'])
    food_ctrl, service_ctrl, tip_ctrl = setup_fuzzy_system(chromosome)
    predicted_tip = execute_fuzzy_inference(food_ctrl, service_ctrl, tip_ctrl, inputs)

    error = abs(actual_tip - predicted_tip)
    total_error += error

  return total_error


def fitness_test(chromosome):
  total_error = 0
  for index, row in test_data.iterrows():
    inputs = {
      'temperature': float(row['food temperature']),
      'flavor': float(row['food flavor']),
      'portion_size': float(row['portion size']),
      'attentiveness': float(row['attentiveness']),
      'friendliness': float(row['friendliness']),
      'speed_of_service': float(row['speed of service'])
    }
    actual_tip = float(row['tip'])
    food_ctrl, service_ctrl, tip_ctrl = setup_fuzzy_system(chromosome)
    predicted_tip = execute_fuzzy_inference(food_ctrl, service_ctrl, tip_ctrl, inputs)

    error = abs(actual_tip - predicted_tip)
    total_error += error

  return total_error


# Training
def generate_chromosome():
  chromosome = []
  arr = sorted([random.randint(0,9) for i in range(3)])
  chromosome += [0, 0, arr[0]] + [0, arr[1], 10] + [arr[2], 10, 10]
  return chromosome

ga = EasyGA.GA()
ga.chromosome_length = 11
ga.gene_impl = lambda: generate_chromosome()
ga.population_size = 2
ga.target_fitness_type = 'min'
ga.generation_goal = 2
ga.fitness_function_impl = fitness_train

ga.evolve()
print(ga.population)
ga.print_best_chromosome()
best_chromosome = ga.population[0]

print("Test data result: ", fitness_test(best_chromosome))
print()
# print("Unoptimized Test data result: ", fitness_old(best_chromosome))