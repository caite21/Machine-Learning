!pip install scikit-fuzzy

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def Pipeline(inputs):

  # input vars as Antecedent
  temperature = ctrl.Antecedent(np.linspace(0, 10, 11),'temperature')
  flavor = ctrl.Antecedent(np.linspace(0, 10, 11),'flavor')
  portion_size = ctrl.Antecedent(np.linspace(0, 10, 11),'portion_size')
  attentiveness = ctrl.Antecedent(np.linspace(0, 10, 11),'attentiveness')
  friendliness = ctrl.Antecedent(np.linspace(0, 10, 11),'friendliness')
  speed_of_service = ctrl.Antecedent(np.linspace(0, 10, 11),'speed_of_service')

  # membership functions for 'poor', 'average', and 'good'
  temperature.automf(3)
  flavor.automf(3)
  portion_size.automf(3)
  attentiveness.automf(3)
  friendliness.automf(3)
  speed_of_service.automf(3)


  # output vars as Consequent
  service_quality = ctrl.Consequent(np.linspace(0, 10, 11), 'service_quality')
  food_quality = ctrl.Consequent(np.linspace(0, 10, 11), 'food_quality')
  tip = ctrl.Consequent(np.linspace(0, 25, 26), 'tip')

  food_quality.automf(3)
  service_quality.automf(3)
  tip.automf(3)

  # all fuzzy rules
  # if any are poor: poor, if any are avg: avg, if flavor is good: good
  rule1_food = ctrl.Rule(temperature['poor'] | flavor['poor'] | portion_size['poor'], food_quality['poor'])
  rule2_food = ctrl.Rule(temperature['average'] | flavor['average'] | portion_size['average'], food_quality['average'])
  rule3_food = ctrl.Rule(flavor['good'], food_quality['good'])
  # if flavor is good but others are bad: avg
  rule4_food = ctrl.Rule(temperature['poor'] & flavor['good'] & portion_size['poor'], food_quality['average'])
  rule5_food = ctrl.Rule(temperature['good'] & flavor['good'] & portion_size['poor'], food_quality['average'])
  rule6_food = ctrl.Rule(temperature['poor'] & flavor['good'] & portion_size['good'], food_quality['average'])

  rule1_service = ctrl.Rule(attentiveness['poor'] | friendliness['poor'] | speed_of_service['poor'], service_quality['poor'])
  rule2_service = ctrl.Rule(attentiveness['average'] | friendliness['average'] | speed_of_service['average'], service_quality['average'])
  rule3_service = ctrl.Rule(friendliness['good'], service_quality['good'])
  rule4_service = ctrl.Rule(attentiveness['poor'] & friendliness['good'] & speed_of_service['poor'], service_quality['average'])
  rule5_service = ctrl.Rule(attentiveness['good'] & friendliness['good'] & speed_of_service['poor'], service_quality['average'])
  rule6_service = ctrl.Rule(attentiveness['poor'] & friendliness['good'] & speed_of_service['good'], service_quality['average'])



  # SIM 1
  # controller
  food_ctrl = ctrl.ControlSystem([rule1_food, rule2_food, rule3_food, rule4_food, rule5_food, rule6_food])
  food_sim = ctrl.ControlSystemSimulation(food_ctrl)

  # pass user inputs
  food_sim.input['temperature'] = inputs['temperature']
  food_sim.input['flavor'] = inputs['flavor']
  food_sim.input['portion_size'] = inputs['portion_size']

  # fuzzy inference and defuzzification
  food_sim.compute()

  # SIM 2
  # controller
  service_ctrl = ctrl.ControlSystem([rule1_service, rule2_service, rule3_service, rule4_service, rule5_service, rule6_service])
  service_sim = ctrl.ControlSystemSimulation(service_ctrl)

  # pass user inputs
  service_sim.input['attentiveness'] = inputs['attentiveness']
  service_sim.input['friendliness'] = inputs['friendliness']
  service_sim.input['speed_of_service'] = inputs['speed_of_service']

  # fuzzy inference and defuzzification
  service_sim.compute()

  # SIM 3
  # controller
  food_quality = ctrl.Antecedent(np.linspace(0, 10, 11), 'food_quality')
  service_quality = ctrl.Antecedent(np.linspace(0, 10, 11), 'service_quality')
  food_quality.automf(3)
  service_quality.automf(3)

  rule1_tip = ctrl.Rule(food_quality['poor'] | service_quality['poor'], tip['poor'])
  rule2_tip = ctrl.Rule(food_quality['average'] & service_quality['average'], tip['average'])
  rule3_tip = ctrl.Rule(food_quality['good'] & service_quality['average'], tip['average'])
  rule4_tip = ctrl.Rule(food_quality['average'] & service_quality['good'], tip['good'])
  rule5_tip = ctrl.Rule(food_quality['good'] & service_quality['good'], tip['good'])


  tip_ctrl = ctrl.ControlSystem([rule1_tip, rule2_tip, rule3_tip, rule4_tip, rule5_tip])
  tip_sim = ctrl.ControlSystemSimulation(tip_ctrl)

  # pass inputs
  tip_sim.input['food_quality'] = food_sim.output['food_quality']
  tip_sim.input['service_quality'] = service_sim.output['service_quality']

  # fuzzy inference and defuzzification
  tip_sim.compute()

  # output value
  return tip_sim.output['tip']


# Get User Input and pass it to Pipeline
def get_continue_input():
  while True:
    # prompted to enter another set of measures
    user_continue = input("Want to enter values? Type yes or no: ")
    if (user_continue == "yes" or user_continue == "no"):
      return user_continue
    else:
      print("Input must be yes or no. Try again.")


def get_numerical_input(prompt):
  while True:
    try:
      num = float(input(prompt + ": "))
      assert(num <= 10 and num >= 0)
      return num
    except:
       # validate inputs (out of range or illegal values)
      print("Input must be a number between 0 and 10. Try again.")

user_continue = True
inputs = {'temperature' : 0,
          'flavor' : 0,
          'portion_size' : 0,
          'attentiveness' : 0,
          'friendliness' : 0,
          'speed_of_service' : 0,}
          
while (True):
  user_continue = get_continue_input()
  if (user_continue == "no"):
    # terminate when the user does not wish to enter any more values
    break
  else:
    for attribute in list(inputs.keys()):
      # user should be able to input values for the six measures
      inputs[attribute] = get_numerical_input("How was the " + attribute + " from 0 to 10")

    # receive a recommended tip (as a percentage of the final bill)
    print("Recommended tip: " + str(Pipeline(inputs)) + "\n")

