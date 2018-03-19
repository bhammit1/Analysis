from __future__ import division
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt,floor,sin
from functions_basic import calc_RMSE

from deap import base, creator, tools

"""
/*******************************************************************
Wiedemann 99 functions used to process and analyze CF Data. 

Author: Britton Hammit
E-mail: bhammit1@gmail.com
Date: 01-26-2018
********************************************************************/
"""

### Wiedemann Calbration Functions ###
def run_w99_GA(cf_collections, cxpb, mutpb, m_indpb, ngen, npop, logfile, figfile=None):
    """
    Main function for running the Gipps Genetic algorithm.
    :param cf_collections: list of (Instance of Processed Data Class with vehicle trajectory data)
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: Number of generations
    :param npop: Number of individuals in the population
    :param logfile: Log file for recording calibration details about each generation
    :param figfile: Figure file for plotting calibration convergence
    :return: [best_score, best_indiv]
    """

    # Set up GA Structure:
    # http://deap.gel.ulaval.ca/doc/default/overview.html
    creator.create(name="FitnessMin", base=base.Fitness, weights=(-1.0,))
    creator.create(name="Individual", base=list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Ranges for integers that will be divided by 10.
    # Ranges from w99 demo unless otherwise noted http://w99demo.com/
    """
    # Done before 2/14/2018 - Sensitivity Analyses
    cc0_low, cc0_high = 0, 100  # w99demo suggested 0.5-5
    cc1_low, cc1_high = 5, 25  # Hitting 5
    cc2_low, cc2_high = 10, 100
    cc3_low, cc3_high = -200, -50  # Hitting -200
    cc4_low, cc4_high = -30, -1  # Hitting -1 (dec to 0?)
    cc5_low, cc5_high = 1, 30  # Hitting 30
    cc6_low, cc6_high = 10, 100  # Hitting 10, close to 100
    cc7_low, cc7_high = 0, 10  # Always hitting 10
    cc8_low, cc8_high = 5, 50  # Hitting 5 & 50 # According to w99 code, should be bounded by maximum cc8
    cc9_low, cc9_high = 5, 80  # Hitting 5
    v_des_low, v_des_high = 1, 400
    car_seed_low, car_seed_high = 1, 1000
    """
    cc0_low, cc0_high = 1, 100  # same as gipps/idm
    cc1_low, cc1_high = 1, 50  # tgap idm
    cc2_low, cc2_high = 1, 150
    cc3_low, cc3_high = -270, -50
    cc4_low, cc4_high = -50, 0
    cc5_low, cc5_high = 0, 50
    cc6_low, cc6_high = 1, 110
    cc7_low, cc7_high = 0, 70
    cc8_low, cc8_high = 1, 70
    cc9_low, cc9_high = 1, 80
    v_des_low, v_des_high = 1, 400
    car_seed_low, car_seed_high = 1, 1000

    # Ranges from original w99 code
    toolbox.register("attr_cc0", random.randint, cc0_low, cc0_high)
    toolbox.register("attr_cc1", random.randint, cc1_low, cc1_high)
    toolbox.register("attr_cc2", random.randint, cc2_low, cc2_high)
    toolbox.register("attr_cc3", random.randint, cc3_low, cc3_high)
    toolbox.register("attr_cc4", random.randint, cc4_low, cc4_high)
    toolbox.register("attr_cc5", random.randint, cc5_low, cc5_high)
    toolbox.register("attr_cc6", random.randint, cc6_low, cc6_high)
    toolbox.register("attr_cc7", random.randint, cc7_low, cc7_high)
    toolbox.register("attr_cc8", random.randint, cc8_low, cc8_high)
    toolbox.register("attr_cc9", random.randint, cc9_low, cc9_high)
    toolbox.register("attr_v_des", random.randint, v_des_low, v_des_high)
    toolbox.register("attr_car_seed", random.randint, car_seed_low, car_seed_high)

    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_cc0, toolbox.attr_cc1,
                                                                         toolbox.attr_cc2, toolbox.attr_cc3,
                                                                         toolbox.attr_cc4, toolbox.attr_cc5,
                                                                         toolbox.attr_cc6, toolbox.attr_cc7,
                                                                         toolbox.attr_cc8, toolbox.attr_cc9,
                                                                         toolbox.attr_v_des, toolbox.attr_car_seed), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # GA
    cx_indpb = 0.5  # percent of the individual that will be switched -- common to use 0.5 https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_and_half_uniform

    low = [cc0_low,cc1_low,cc2_low,cc3_low,cc4_low,cc5_low,cc6_low,cc7_low,cc8_low,cc9_low,v_des_low]
    up = [cc0_high,cc1_high,cc2_high,cc3_high,cc4_high,cc5_high,cc6_high,cc7_high,cc8_high,cc9_high,v_des_high]

    toolbox.register("mate", tools.cxUniform, indpb=cx_indpb)
    toolbox.register("mutate", w99_mutUniformInt, low=low, up=up, indpb=m_indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_w99_GA, cf_collections=cf_collections)

    pop = toolbox.population(n=npop)

    log, best_score, best_indiv = evolve_w99_GA(population=pop, toolbox=toolbox, cxpb=cxpb, mutpb=mutpb,
                                                m_indpb=m_indpb, ngen=ngen, logfile=logfile)

    if figfile is not None:
        plt.plot(log['gen'], log['min_score'], label="{} {} {} {}".format(cxpb, mutpb, ngen, npop))
        plt.xlabel("Generation")
        plt.ylabel("Min Fitness of Population")
        plt.legend(loc="upper right")
        plt.savefig(figfile)

    return best_score, best_indiv


def evolve_w99_GA(population, toolbox, cxpb, mutpb, m_indpb, ngen, logfile):
    """
    Evolve a population through the DEAP GA.
    Algorithm altered from eaSimple, provided as part of the DEAP algorithms.py
    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: Number of generations
    :param logfile: Log file for recording calibration details about each generation
    :return: [log, best_score, best_indiv], where the log is a dictionary containing minimum
                scores and best individuals for each generation, the best score, and the best individual
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    initiate_w99_calibration_log_file(file=logfile, cxpb=cxpb, mutpb=mutpb, m_indpb=m_indpb, pop_size=len(population),ngen=ngen)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof_global = tools.HallOfFame(1)
    hof_global.update(population)

    record = stats.compile(population) if stats else {}

    append_to_w99_calibration_log_file(file=logfile, gen=0, no_unique_indiv=len(invalid_ind), min_score=record['min'],
                                                  ave_score=record['avg'], max_score=record['max'],
                                                  std_score=record['std'], best_indiv=hof_global[0])

    log = {}
    log['min_score'] = list()
    log['gen'] = list()
    log['hof_local'] = list()
    log['hof_global'] = list()

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring_a = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = [toolbox.clone(ind) for ind in offspring_a]
        del offspring_a

        # Apply crossover and mutation on the offspring
        # Changed it so that mutation occurs first - then crossover... this way more individuals are impacted

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values, offspring[i].fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the global HOF to track the best individual seen in the evolution process
        hof_global.update(offspring)
        # Create new HOF to track best individual in this generation.
        hof_local = tools.HallOfFame(1)
        hof_local.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring
        del offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}  # record is a dictionary of the operator and value

        append_to_w99_calibration_log_file(file=logfile, gen=gen, no_unique_indiv=len(invalid_ind), min_score=record['min'],
                                                      ave_score=record['avg'], max_score=record['max'],
                                                      std_score=record['std'], best_indiv=hof_local[0])
        log['gen'].append(gen)
        log['min_score'].append(record['min'])
        log['hof_local'].append(hof_local[0])
        log['hof_global'].append(hof_global[0])

        del hof_local

    best_score = toolbox.evaluate(hof_global[0])[0]
    best_indiv = hof_global[0]

    append_to_w99_calibration_log_file(file=logfile, gen=gen, no_unique_indiv=len(invalid_ind),
                                                  min_score= best_score,
                                                  ave_score=0, max_score=0,
                                                  std_score=0, best_indiv=best_indiv)

    return log, best_score, best_indiv


def evaluate_w99_GA(individual,cf_collections):
    """
    Generate the score for an individual.
        According to literature, it is best to calibrate a model based on the RMSE of spacing;
        therefore, the RMSE_dX was chosen as the fitness function used to evaluate the score
        of each individual.
    :param individual: Array of model parameters: [t_rxn, V_des, a_des, d_des, d_lead, g_min]
    :param cf_collections: list of (Instance of Processed Data Class with vehicle trajectory data)
    :return: Individual Score = RMSE_dX: Float value indicating the individual score
    """

    """
    # Default Parameters - Sanity Check
    cc0 = 1.5*10  # standstill distance [m]
    cc1 = 1.3*10  # spacing time [s]
    cc2 = 4.0*10  # following variation ("max drift") [m]
    cc3 = -12.0*10  # threshold for entering 'following' [s]
    cc4 = -0.25*10  # negative 'following' threshold [m/s]
    cc5 = 0.35*10  # positive 'following' threshold [m/s]
    cc6 = 6.0*10  # speed dependency of oscillation [10^-4 RAD/s]
    cc7 = 0.25*10  # oscillation acceleration [m/s2]
    cc8 = 2.0*10  # standstill acceleration [m/s2]
    cc9 = 1.5*10  # acceleration at 80kph [m/s2]
    v_des = 30*10
    car_seed = 10
    individual = [cc0,cc1,cc2,cc3,cc4,cc5,cc6,cc7,cc8,cc9,v_des,car_seed]
    """

    RMSE_list = list()
    timestep = 0.1  # seconds

    for collection in cf_collections:  # Loop through each car-following collection (event) in the list
        dX_pred_list = list()
        dX_act_list = list()

        # For sanity checking:
        vfoll_pred_list = list()
        vfoll_act_list = list()
        vlead_act_list = list()
        afoll_pred_list = list()
        afoll_act_list = list()
        alead_act_list = list()

        # Initialize v_foll, a_foll, and dX values
        vfoll_pred_list.append(collection.v_foll[0])
        afoll_pred_list.append(collection.a_foll[0])
        dX_pred_list.append(collection.dX[0])
        vfoll_act_list.append(collection.v_foll[0])
        afoll_act_list.append(collection.a_foll[0])
        dX_act_list.append(collection.dX[0])
        # Initialize to get the first "following code"
        code_foll = 'A'  # Don't think it matters what it is initialized as!

        for index in xrange(len(collection.dX)-1):
            i = index + 1
            # Pull Actual Variables:
            v_lead = collection.v_lead[i-1]
            a_lead = collection.a_lead[i-1]

            v_foll = vfoll_pred_list[i-1]
            a_foll = afoll_pred_list[i-1]
            dX = dX_pred_list[i-1]

            # Predict Following Vehicle Acceleration
            a_foll_new,code_foll = w99_predict_a_f(individual=individual,v_foll=v_foll,v_lead=v_lead,a_foll=a_foll,
                                         a_lead=a_lead,dX=dX,code_foll=code_foll)

            # Calculate New Velocity & Spacing for the next timestamp
            v_foll_new = v_foll + a_foll_new*timestep
            d_lead = (collection.v_lead[i-1]+collection.v_lead[i])/2*timestep  # distance traveled by lead vehicle
            d_foll = (v_foll+v_foll_new)/2*timestep  # distance traveled by following vehicle
            dX_new = dX - d_foll + d_lead

            """
            # Crash Penalty
            if dX_new < 0:  # severe penalty if crash occurs.
                dX_new = np.inf
            """

            # Initialize with predictions from [0] and actual variables from [1]
            dX_pred_list.append(dX_new)  # This iteration
            dX_act_list.append(collection.dX[i])  # The future "actual" value

            # Copy velocity and acceleration to lists for sanity checks
            vfoll_pred_list.append(v_foll_new)
            vfoll_act_list.append(collection.v_foll[i])
            vlead_act_list.append(collection.v_lead[i])
            afoll_pred_list.append(a_foll_new)
            afoll_act_list.append(collection.a_foll[i])
            alead_act_list.append(collection.a_lead[i])

        RMSE_dx = calc_RMSE(dX_pred_list, dX_act_list)
        RMSE_list.append(RMSE_dx)

        # Sanity Check - Plot dX,v,a Values
        """
        f, (ax1,ax2,ax3) = plt.subplots(3)
        ax1.plot(dX_act_list,'r')
        ax1.plot(dX_pred_list,'b')
        ax1.set_title('Spacing')
        ax2.plot(vfoll_act_list,'r')
        ax2.plot(vfoll_pred_list,'b')
        ax2.plot(vlead_act_list, 'g')
        ax2.set_title('Velocity')
        ax3.plot(afoll_act_list,'r', label='actual')
        ax3.plot(afoll_pred_list, 'b', label='predicted')
        ax3.plot(alead_act_list, 'g', label='lead')
        ax3.set_title('Acceleration')
        ax3.legend()
        f.subplots_adjust(hspace=0.5)
        f.suptitle('Sanity Check, Individual = {}, RMSE = {}'.format(individual,RMSE_dx))
        plt.show()
        """

    # Take a weighted average of the RMSE from each CF event
    index = 0
    length_list = list()
    for collection in cf_collections:
        length_list.append(float(collection.point_count()))
        index += 1
    RMSE_list_weighted = list()
    for i in range(len(cf_collections)):
        factor = length_list[i] / np.sum(length_list)
        RMSE_list_weighted.append(RMSE_list[i] * factor)

    RMSE_all = np.sum(RMSE_list_weighted)

    return RMSE_all,


def w99_predict_a_f(individual,v_foll,v_lead,a_foll,a_lead,dX,code_foll):
    ### Model Parameters ###
    cc0,cc1,cc2,cc3,cc4,cc5,cc6,cc7,cc8,cc9,v_des,car_seed = individual
    cc0 = cc0/10.
    cc1 = cc1/10.
    cc2 = cc2/10.
    cc3 = cc3/10.
    cc4 = cc4/10.
    cc5 = cc5/10.
    cc6 = cc6/10.
    cc7 = cc7/10.
    cc8 = cc8/10.
    cc9 = cc9/10.
    v_des = v_des/10.
    """
    # Default Parameters
    cc0 = 1.5  # standstill distance [m]
    cc1 = 1.3  # spacing time [s]
    cc2 = 4.0  # following variation ("max drift") [m]
    cc3 = -12.0  # threshold for entering 'following' [s]
    cc4 = -0.25  # negative 'following' threshold [m/s]
    cc5 = 0.35  # positive 'following' threshold [m/s]
    cc6 = 6.0  # speed dependency of oscillation [10^-4 RAD/s]
    cc7 = 0.25  # oscillation acceleration [m/s2]
    cc8 = 2.0  # standstill acceleration [m/s2]
    cc9 = 1.5  # acceleration at 80kph [m/s2]
    """

    # Unit Adjustment
    cc6 = cc6 / 10000
    timestep = 0.1

    ### Vehicle Inputs ###
    """
    car_seed = 1  # random seed for driving aggressiveness
    v_des = 13  # desired speed [m/s]
    """
    ### Calculation Inputs ###
    """
    dX = 1  # leader position - follower position - lead vehicle length [m]
    v_foll = 15  # [m/s]
    a_foll = 1  # [m/s2]
    v_lead = 10  # [m/s]
    a_lead = 1  # [m/s2]
    code_foll = 'w'  # past regime: 'f','w','A','B'
    """

    dV = v_lead - v_foll  # leader - follower [m/s]

    ### Framework Calculations ###
    if v_lead <= 0:
        sdxc = cc0  # minimum following distance at a stop!
    else:
        if dV >= 0 or a_lead < -1:
            v_slower = v_foll
        else:
            v_slower = v_lead + dV * (simpleRandom(seed=car_seed) - 0.5)
        sdxc = cc0 + cc1 * v_slower  # minimum following distance considered safe by driver

    sdxo = cc2 + sdxc  # maximum following distance (upper limit car-following process)

    sdxv = sdxo + cc3 * (dV - cc4)

    sdv = cc6 * dX * dX  # distance driver starts perceiving speed differences when approaching slower leader

    if v_lead > 0:
        sdvc = cc4 - sdv  # minimum closing dV
    else:
        sdvc = 0  # minimum closing dV

    if v_foll > cc5:
        sdvo = cc5 + sdv  # minimum opening dV
    else:
        sdvo = sdv  # minimum opening dV

    ### Status Variables ###
    follower_status = {}  # dict with information related to following vehicle
    follower_status['dX'] = dX
    follower_status['dV'] = dV
    follower_status['sdxc'] = sdxc
    follower_status['sdxv'] = sdxv
    follower_status['sdxo'] = sdxo
    follower_status['sdvc'] = sdvc
    follower_status['sdvo'] = sdvo

    ### Calculate Acceleration Behavior ###
    a_foll_new = 0  # m/s2

    ## Regime A ##
    if dX <= sdxc and dV <= sdvo:
        follower_status['description'] = 'Decelerate - Increase Distance'
        follower_status['message_condition'] = 'Too Close'
        follower_status['message_action'] = 'Decelerate'
        follower_status['code'] = 'A'

        if v_foll > 0:
            if dV < 0:
                if dX > cc0:
                    a_foll_new = min([a_lead + (dV * dV) / (cc0 - dX), a_foll])
                else:
                    a_foll_new = min([a_lead + 0.5 * (dV - sdvo), a_foll])
                if a_foll_new > -cc7:
                    a_foll_new = -cc7
                else:
                    a_foll_new = max([a_foll_new, -10 + 0.5 * sqrt(v_foll)])  # Units??

    ## Regime "B" ##
    elif dV < sdvc and dX < sdxv:
        follower_status['description'] = 'Decelerate - Decrease Distance'
        follower_status['message_condition'] = 'Too Close'
        follower_status['message_action'] = 'Decelerate'
        follower_status['code'] = 'B'

        a_foll_new = max([0.5 * dV * dV / (-dX + sdxc - 0.1), -10])  # feasible acceleration capped by g and ability to stop

    ## Regime "f" ##
    elif dV < sdvo and dX < sdxo:
        follower_status['description'] = 'Accelerate/Decelerate - Keep Distance'
        follower_status['message_condition'] = 'Keep Distance'
        follower_status['message_action'] = 'Follow'
        follower_status['code'] = 'f'

        if a_foll <= 0:
            a_foll_new = min([a_foll, -cc7])
        else:
            a_foll_new = max([a_foll, cc7])
            #a_foll_new = min([a_foll_new, v_des - v_foll])  # W99 & Demo... units??
            #a_foll_new = min([a_foll_new, (v_des - v_foll)/timestep, cc8])  # Attempt to correct units problem and cap the acceleration to reasonable value.
            a_foll_new = min([a_foll_new, (v_des - v_foll) / timestep])  # corrected timestep problem for units but no restrictions on acceleration

    ## Regime "w" ##
    else:
        follower_status['description'] = 'Accelerate/Relax - Increase or Keep Speed'
        follower_status['message_condition'] = 'Free Flow'
        follower_status['message_action'] = 'Accelerate'
        follower_status['code'] = 'w'

        if dX > sdxc:
            if code_foll == 'w':
                a_foll_new = cc7
            else:
                a_max = cc8 + cc9 * min([v_foll, 80 * 1000 / 3600]) + simpleRandom(seed=car_seed)  # capped at 80km/hr
                if dX < sdxo:
                    a_foll_new = min([dV * dV / (sdxo - dX), a_max])
                else:
                    a_foll_new = a_max
            # a_foll_new = min([a_foll_new, v_des - v_foll])  # W99 & Demo... units??
            # a_foll_new = min([a_foll_new, (v_des - v_foll) / timestep,cc8])  # Attempt to correct units problem and cap the acceleration to reasonable value.
            a_foll_new = min([a_foll_new, (v_des - v_foll) / timestep])
            if abs(v_des - v_foll) < 0.1:
                follower_status['message_action'] = 'Top Speed'

    """
    print "Acceleration: {}".format(a_foll_new)
    for attribute, value in follower_status.items():
        print('    {} : {}'.format(attribute, value))
    
    if abs(a_foll_new)>1:
        print "{}: {}".format(a_foll_new,follower_status['code'])
    """
    return a_foll_new,follower_status['code']


def w99_mutUniformInt(individual, low, up, indpb):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by a integer uniformly drawn between *low* and *up* inclusively.

    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)-1  # car seed is not going to be mutated!!

    for i, xl, xu in zip(xrange(size), low, up):
        if random.random() < indpb:
            individual[i] = random.randint(xl, xu)

    return individual,


def simpleRandom(seed):
    x = sin(seed)*10000
    return x - floor(x)



### Log Files ###
def initiate_w99_calibration_log_file(file,cxpb,mutpb,m_indpb,pop_size,ngen):

    file.write('Wiedemann CFM Calibration - DEAP GA Implementation')
    file.write('\n')
    file.write('cxpb,mutpb,m_indpb,pop_size,ngen')
    file.write('\n')
    file.write('{},{},{},{},{}'.format(cxpb,mutpb,m_indpb,pop_size,ngen))
    file.write('\n')
    file.write('Gen,No Unique Indiv,Min Score,Ave Score,Max Score,Std Score,')
    file.write('cc0,cc1,cc2,cc3,cc4,cc5,cc6,cc7,cc8,cc9,v_des,car_seed')
    file.write('\n')

    print 'Wiedemann CFM Calibration - DEAP GA Implementation'
    print 'cxpb: {} | mutpb: {} | m_indpb: {} | pop_size: {} | ngen: {}'.format(cxpb,mutpb,m_indpb,pop_size,ngen)
    print '%4s | %4s | %8s | %8s | %8s | %8s | %5s, %5s, %5s, %5s, %5s, %5s, %5s, %5s, %5s, %5s, %5s, %5s' % (
    'gen', 'cnt', 'min', 'ave', 'max', 'std', 'cc0', 'cc1', 'cc2', 'cc3', 'cc4', 'cc5', 'cc6', 'cc7', 'cc8', 'cc9',
    'v_d', 'seed')


def append_to_w99_calibration_log_file(file,gen,no_unique_indiv,min_score,ave_score,max_score,std_score,best_indiv):

    file.write('{},{},{},{},{},{},'.format(gen,no_unique_indiv,min_score,ave_score,max_score,std_score))
    file.write(
        '{},{},{},{},{},{},{},{},{},{},{},{}'.format(best_indiv[0] / 10., best_indiv[1] / 10., best_indiv[2] / 10.,
                                                     best_indiv[3] / 10., best_indiv[4] / 10., best_indiv[5] / 10.,
                                                     best_indiv[6] / 10., best_indiv[7] / 10., best_indiv[8] / 10.,
                                                     best_indiv[9] / 10., best_indiv[10] / 10., best_indiv[11]))

    file.write('\n')

    print '%4.0f | %4.0f | %8.2f | %8.2f | %8.2f | %8.2f | %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.1f, %5.0f' % (
    gen, no_unique_indiv, min_score, ave_score, max_score, std_score, best_indiv[0] / 10., best_indiv[1] / 10.,
    best_indiv[2] / 10., best_indiv[3] / 10., best_indiv[4] / 10., best_indiv[5] / 10., best_indiv[6] / 10.,
    best_indiv[7] / 10., best_indiv[8] / 10., best_indiv[9] / 10., best_indiv[10] / 10., best_indiv[11])


def initiate_w99_calibration_summary_file(file):
    file.write('iteration,time,cxpd,mutpd,m_indpb,ngen,npop,score,cc0,cc1,cc2,cc3,cc4,cc5,cc6,cc7,cc8,cc9,v_des,car_seed')
    file.write('\n')


def append_to_w99_calibration_summary_file(file, elapsed_time, iteration, cxpb, mutpb, m_indpb, ngen, npop,
                                                  score, best_indiv):
    file.write('{},'.format(iteration))
    file.write('{},'.format(elapsed_time))
    file.write('{},{},{},{},{},'.format(cxpb, mutpb, m_indpb, ngen, npop))
    file.write('{},'.format(score))
    file.write(
        '{},{},{},{},{},{},{},{},{},{},{},{}'.format(best_indiv[0] / 10., best_indiv[1] / 10., best_indiv[2] / 10.,
                                                     best_indiv[3] / 10., best_indiv[4] / 10., best_indiv[5] / 10.,
                                                     best_indiv[6] / 10., best_indiv[7] / 10., best_indiv[8] / 10.,
                                                     best_indiv[9] / 10., best_indiv[10] / 10., best_indiv[11]))

    file.write('\n')


def initiate_w99_calibration_cs_summary_file(file):
    file.write('trip_set_no,trip_no,driver_id,adverse_cond,trip_cond,time,score,cc0,cc1,cc2,cc3,cc4,cc5,cc6,cc7,cc8,cc9,v_des,car_seed')
    file.write('\n')


def append_to_w99_calibration_cs_summary_file(file, elapsed_time, trip_set_no, trip_no, driver_id,
                                                            adverse_cond, trip_cond, score, best_indiv):
    file.write('{},'.format(trip_set_no))
    file.write('{},'.format(trip_no))
    file.write('{},{},{},'.format(driver_id, adverse_cond, trip_cond))
    file.write('{},'.format(elapsed_time))
    file.write('{},'.format(score))
    file.write(
        '{},{},{},{},{},{},{},{},{},{},{},{}'.format(best_indiv[0] / 10., best_indiv[1] / 10., best_indiv[2] / 10.,
                                                     best_indiv[3] / 10., best_indiv[4] / 10., best_indiv[5] / 10.,
                                                     best_indiv[6] / 10., best_indiv[7] / 10., best_indiv[8] / 10.,
                                                     best_indiv[9] / 10., best_indiv[10] / 10., best_indiv[11]))
    file.write('\n')


def initiate_201802Calib_w99_summary_file(file):
    file.write('2018-02-15 Wiedemann 99 Calibration Summary File')
    file.write('\n')
    # Trip Info
    file.write('trip_no,total_run_time_sec,')
    file.write('driver_id,total_trip_length_min,total_trip_length_km,')
    file.write('stac_availability,')
    file.write('time_bin,day,month,year,')
    # Car-following
    file.write('time_cf_percent,time_cf_min,no_cf_events,')
    # Demographics
    file.write('gender,age_group,ethnicity,race,education,marital_status,living_status,work_status,')
    file.write('household_population,income,')
    file.write('miles_driven_last_year,')
    # Behavior
    file.write('frequency_tailgating,frequency_disregarding_speed_limit,frequency_aggressive_braking,')
    # Calibration Info
    file.write('calibration_time_sec,calibration_score,cc0,cc1,cc2,cc3,cc4,cc5,cc6,cc7,cc8,cc9,v_des,car_seed')
    file.write('\n')


def append_to_201802Calib_w99_summary_file(file, trip_no, driver_id, point_collection, cf_collections,
                                                  stac_data_available, demographics_data, behavior_data, calib_time,
                                                  calib_score, calib_best_indiv, total_time):
    # Trip Info
    file.write('{},{},'.format(trip_no,total_time))
    file.write('{},{},{},'.format(driver_id, point_collection.time_elapsed() / 60,
                                     point_collection.dist_traveled()))
    file.write("{},".format(stac_data_available))
    time1, day, month, year = point_collection.time_day_month_year()
    file.write("{},{},{},{},".format(time1, day, month, year))

    # Car-following
    file.write("{},{},{},".format(point_collection.percent_car_following(), point_collection.time_car_following(),
                                        len(cf_collections)))

    # Driver Demographics
    if demographics_data == None:
        for j in range(11):
            file.write('{},'.format(np.nan))
    else:
        # file.write('Gender,Age Group,Ethnicity,Race,Education,Marital Status,Living Status,Work Status,')
        file.write('{},{},{},'.format(demographics_data[1], demographics_data[2], demographics_data[3]))
        file.write('{},{},{},'.format(demographics_data[4], demographics_data[6], demographics_data[7]))
        file.write('{},{},'.format(demographics_data[8], demographics_data[10]))
        # file.write('Income,Household Population,')
        file.write('{},{},'.format(demographics_data[11], demographics_data[12]))
        # file.write('Miles Driven Last Year,')
        file.write('{},'.format(demographics_data[44]))

    # Driver Behavior
    if behavior_data == None:
        file.write("{},{},{},".format(np.nan, np.nan, np.nan))
    else:
        # file.write('Frequency of Tailgating,Frequency of Disregarding Speed Limit,Frequency of Aggressive Braking')
        file.write('{},{},{},'.format(behavior_data[3], behavior_data[12], behavior_data[24]))

    # Calibration
    file.write('{},'.format(calib_time))
    file.write('{},'.format(calib_score))
    file.write(
        '{},{},{},{},{},{},{},{},{},{},{},{}'.format(calib_best_indiv[0] / 10., calib_best_indiv[1] / 10., calib_best_indiv[2] / 10.,
                                                     calib_best_indiv[3] / 10., calib_best_indiv[4] / 10., calib_best_indiv[5] / 10.,
                                                     calib_best_indiv[6] / 10., calib_best_indiv[7] / 10., calib_best_indiv[8] / 10.,
                                                     calib_best_indiv[9] / 10., calib_best_indiv[10] / 10., calib_best_indiv[11]))


    file.write("\n")


#todo - Update this to W99 Parameters.
### Plotting/Analysis Functions ###
def w99_sensitivity_analysis_plot(summary_file, date, save_path, CXPB, MUTPB, NGEN, NPOP):
    df = pd.read_csv(filepath_or_buffer=summary_file, delimiter=',', header=0)

    # Score Plot
    fig, axes = plt.subplots(nrows=len(CXPB), ncols=len(MUTPB), figsize=(15, 12))  # figsize=(13,11)
    fig.suptitle('W99 Calibration Sensitivity Analysis | {} Generations | {}'.format(NGEN, date), fontsize=16,
                 fontweight='bold')
    for i in range(len(CXPB)):
        for j in range(len(MUTPB)):
            # Create data frame for specific plot
            df_temp = df[(df.cxpd == CXPB[i]) & (df.mutpd == MUTPB[j])]
            no_iterations = len(df_temp[df_temp.npop == NPOP[0]])
            df_temp.plot(x='npop', y='score', kind='scatter', subplots=True, ax=axes[i, j],
                         label='{} Iterations'.format(no_iterations), color='b')
            axes[i, j].set_title('cxpb: {} | mutpb: {}'.format(CXPB[i], MUTPB[j]), fontweight='bold')

            # Y Limits
            axes[i, j].set_ylim([0.1, 0.35])  # Vehicle 13 & 41
            # axes[i,j].set_ylim([0.45,0.7])  # Vehicle 35
            axes[i, j].set_ylabel('Score: RMSE of dX [m]', fontsize=12)

            # X Limits
            min_pop = min(NPOP)
            max_pop = max(NPOP)
            diff_pop = max_pop - min_pop
            buffer_dist = diff_pop * 0.2 / 0.6  # 20% buffer on each side
            axes[i, j].set_xlim([min_pop - buffer_dist, max_pop + buffer_dist])
            axes[i, j].set_xlabel('Population Size', fontsize=12)
            del buffer_dist

            # Averages & Standard Deviations Per Population
            ave_list = list()  # list of average scores
            std_list = list()  # list of std of scores
            for npop in NPOP:
                df_temp2 = df_temp[df_temp.npop == npop]
                ave_list.append(np.nanmean(df_temp2.score))
                std_list.append(np.nanstd(df_temp2.score))
            # Horizontal line for each Average
            x_buffer_dist = diff_pop * 0.15 / 0.7  # 10% buffer on each side
            y_buffer_dist = (0.35 - 0.1) * 0.02 / 0.96  # 2% buffer above
            bbox = dict(boxstyle="round,pad=0.1", fc='white', ec='white', lw=0, alpha=0.8)
            for k in range(len(NPOP)):
                axes[i, j].plot((NPOP[k] - x_buffer_dist, NPOP[k] + x_buffer_dist), (ave_list[k], ave_list[k]),
                                color='r', linestyle='--', label='ave')
                axes[i, j].annotate(('Ave: {:4.3f}'.format(ave_list[k])),
                                    xy=(NPOP[k] - x_buffer_dist, ave_list[k] + y_buffer_dist),
                                    xytext=(NPOP[k] - x_buffer_dist, ave_list[k] + y_buffer_dist), color='g', bbox=bbox,
                                    fontweight='bold')
                axes[i, j].annotate(('Std: {:4.3f}'.format(std_list[k])),
                                    xy=(NPOP[k] - x_buffer_dist, ave_list[k] - y_buffer_dist * 2.5),
                                    xytext=(NPOP[k] - x_buffer_dist, ave_list[k] - y_buffer_dist * 2.5), color='g',
                                    bbox=bbox, fontweight='bold')

            del min_pop, max_pop, diff_pop, npop, ave_list, std_list, k, x_buffer_dist, y_buffer_dist, bbox

            del df_temp

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    plt.subplots_adjust(top=0.92, bottom=0.08)
    # plt.show()

    fig.savefig(os.path.join(save_path, '{}'.format(date)))


def w99_convergence_plot(log_file, date, save_path):
    df = pd.read_csv(filepath_or_buffer=log_file, delimiter=',', header=3)
    convergence_variables = ['Min Score', 'Best: T_rxn', 'Best: V_des', 'Best: a_des', 'Best: d_des', 'Best: d_lead',
                             'Best: g_min']
    # Score Plot
    fig, axes = plt.subplots(nrows=len(convergence_variables), ncols=1, figsize=(12, 16))  # figsize=(13,11)
    fig.suptitle('Gipps Calibration Convergence | {}'.format(date), fontsize=16, fontweight='bold')

    plot_pos_index = 0
    for var in convergence_variables:
        # Create data frame for specific plot
        if var == 'Min Score':
            color = 'darkgreen'
        elif var == 'Best: d_des':
            color = 'maroon'
        else:
            color = 'navy'
        df.plot(x='Gen', y=var, kind='line', subplots=True, ax=axes[plot_pos_index], color=color,
                label='value by generation')
        axes[plot_pos_index].set_title('{}'.format(var))
        axes[plot_pos_index].legend(loc='lower right')

        # Identify Generation with Last Score Change
        last_change = 0
        for j in range(len(df['Gen']) - 1):
            if df[var][j] != df[var][j + 1]:
                last_change = df['Gen'][j + 1]

        axes[plot_pos_index].axvline(last_change, color='r', linestyle='--', label='last change')

        # X Limits
        axes[plot_pos_index].set_xlabel('')

        plot_pos_index += 1

    plt.tight_layout()
    plt.subplots_adjust(hspace=.8, wspace=0.2)
    plt.subplots_adjust(top=0.92, bottom=0.03)
    # plt.show()

    fig.savefig(os.path.join(save_path, '{}'.format(date)))


def w99_sensitivity_analysis_file(summary_file, date, save_path, CXPB, MUTPB, M_INDPB, NGEN, NPOP):
    df_summary = pd.read_csv(filepath_or_buffer=summary_file, delimiter=',', header=0)

    # Set up Calibration Summary File
    target = open(os.path.join(save_path, '{}_calibration_summary.csv'.format(date)), 'w')
    target.write('GroupNo,cxpb,mutpb,m_indpb,')
    target.write('ngen,npop,')
    target.write('score_ave,score_std,score_min,')
    target.write('time_ave,time_std,time_min,')
    target.write('last_gen_ave,last_gen_std,last_gen_min,freq_converged')
    target.write('\n')

    group_no = 0  # Counter for each set of GA parameters
    iteration_no = 0  # Counter for each individual iteration
    for i in range(len(CXPB)):
        for j in range(len(MUTPB)):
            for k in range(len(NPOP)):
                for m in range(len(M_INDPB)):
                    for n in range(len(NGEN)):
                        group_no += 1
                        # Create data frame for specific plot
                        df_summary_temp = df_summary[(df_summary.cxpd == CXPB[i]) & (df_summary.mutpd == MUTPB[j]) & (
                        df_summary.npop == NPOP[k]) & (df_summary.m_indpb == M_INDPB[m])]
                        no_iterations = len(df_summary_temp)

                        # Averages & Standard Deviations Per Population -- Scores
                        ave_scores = np.nanmean(df_summary_temp.score)
                        std_scores = np.nanstd(df_summary_temp.score)
                        min_scores = df_summary_temp.score.min()

                        # Averages & Standard Deviations Per Population -- CompTime
                        ave_time = np.nanmean(df_summary_temp.time)
                        std_time = np.nanstd(df_summary_temp.time)
                        min_time = df_summary_temp.time.min()

                        # Looking at Convergence/Generation with the lowest reported MIN score.
                        last_change = list()
                        converge_counter = 0
                        for iteration_no in df_summary_temp.iteration:
                            log_filename = '{}_{}_logfile.csv'.format(date, iteration_no)
                            log_file = open(os.path.join(save_path, log_filename), 'r')
                            df_log = pd.read_csv(filepath_or_buffer=log_file, delimiter=',', header=3)

                            # Identify Generation with Last Score Change
                            for p in range(len(df_log['Gen']) - 1):
                                if df_log['Min Score'][p] != df_log['Min Score'][p + 1]:
                                    last_change_temp = df_log['Gen'][p + 1]
                            last_change.append(last_change_temp)
                            if last_change_temp < NGEN[n]:
                                converge_counter += 1

                        ave_last_change = np.nanmean(last_change)
                        std_last_change = np.nanstd(last_change)
                        min_last_change = np.nanmin(last_change)

                        # Summary Calibration File
                        target.write('{},{},{},{},'.format(group_no, CXPB[i], MUTPB[j], M_INDPB[m]))
                        target.write('{},{},'.format(NGEN[n], NPOP[k]))
                        target.write('{},{},{},'.format(ave_scores, std_scores, min_scores))
                        target.write('{},{},{},'.format(ave_time, std_time, min_time))
                        target.write(
                            '{},{},{},{}'.format(ave_last_change, std_last_change, min_last_change, converge_counter))
                        target.write('\n')
    target.close()
