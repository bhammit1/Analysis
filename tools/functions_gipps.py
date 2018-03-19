from __future__ import division
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from functions_basic import calc_RMSE

from deap import base, creator, tools

"""
/*******************************************************************
Gipps functions used to process and analyze CF Data. 

Author: Britton Hammit
E-mail: bhammit1@gmail.com
Date: 11-01-2017
********************************************************************/
"""

### Gipps Calibration Functions ###
def run_gipps_GA(cf_collections, cxpb, mutpb, m_indpb, ngen, npop, logfile, figfile=None):
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

    t_rxn_min, t_rxn_max = 1, 20
    V_des_min, V_des_max = 1, 400
    a_des_min, a_des_max = 1, 40  # same as idm
    d_des_min, d_des_max = -40, -1  # same as idm
    d_lead_min, d_lead_max = -40, -1
    g_min_min, g_min_max = 1, 100  # same as idm

    toolbox.register("attr_t_rxn", random.randint, t_rxn_min, t_rxn_max)
    toolbox.register("attr_V_des", random.randint, V_des_min, V_des_max)
    toolbox.register("attr_a_des", random.randint, a_des_min, a_des_max)
    toolbox.register("attr_d_des", random.randint, d_des_min, d_des_max)
    toolbox.register("attr_d_lead", random.randint, d_lead_min, d_lead_max)
    toolbox.register("attr_g_min", random.randint, g_min_min, g_min_max)

    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_t_rxn, toolbox.attr_V_des,
                                                                         toolbox.attr_a_des, toolbox.attr_d_des,
                                                                         toolbox.attr_d_lead, toolbox.attr_g_min), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # GA
    cx_indpb = 0.5  # percent of the individual that will be switched -- common to use 0.5 https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_and_half_uniform

    low = [t_rxn_min, V_des_min, a_des_min, d_des_min, d_lead_min, g_min_min]
    up = [t_rxn_max, V_des_max, a_des_max, d_des_max, d_lead_max, g_min_max]

    toolbox.register("mate", tools.cxUniform, indpb=cx_indpb)
    toolbox.register("mutate", tools.mutUniformInt, low=low, up=up, indpb=m_indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_gipps_GA, cf_collections=cf_collections)

    pop = toolbox.population(n=npop)

    log, best_score, best_indiv = evolve_gipps_GA(population=pop, toolbox=toolbox, cxpb=cxpb, mutpb=mutpb,
                                                  m_indpb=m_indpb, ngen=ngen,
                                                  logfile=logfile)

    if figfile is not None:
        plt.plot(log['gen'], log['min_score'], label="{} {} {} {}".format(cxpb, mutpb, ngen, npop))
        plt.xlabel("Generation")
        plt.ylabel("Min Fitness of Population")
        plt.legend(loc="upper right")
        plt.savefig(figfile)

    return best_score, best_indiv


def evolve_gipps_GA(population, toolbox, cxpb, mutpb, m_indpb, ngen, logfile):
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

    initiate_gipps_calibration_log_file(file=logfile, cxpb=cxpb, mutpb=mutpb, m_indpb=m_indpb,
                                             pop_size=len(population), ngen=ngen)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof_global = tools.HallOfFame(1)
    hof_global.update(population)

    record = stats.compile(population) if stats else {}

    append_to_gipps_calibration_log_file(file=logfile, gen=0,
                                              no_unique_indiv=len(invalid_ind), min_score=record['min'],
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

        append_to_gipps_calibration_log_file(file=logfile, gen=gen,
                                                  no_unique_indiv=len(invalid_ind), min_score=record['min'],
                                                  ave_score=record['avg'], max_score=record['max'],
                                                  std_score=record['std'], best_indiv=hof_local[0])
        log['gen'].append(gen)
        log['min_score'].append(record['min'])
        log['hof_local'].append(hof_local[0])
        log['hof_global'].append(hof_global[0])

        del hof_local

    best_score = toolbox.evaluate(hof_global[0])[0]
    best_indiv = hof_global[0]

    append_to_gipps_calibration_log_file(file=logfile, gen=gen,
                                              no_unique_indiv=len(invalid_ind),
                                              min_score=best_score,
                                              ave_score=0, max_score=0,
                                              std_score=0, best_indiv=best_indiv)

    return log, best_score, best_indiv


def evaluate_gipps_GA(individual, cf_collections):
    """
    Generate the score for an individual.
        According to literature, it is best to calibrate a model based on the RMSE of spacing;
        therefore, the RMSE_dX was chosen as the fitness function used to evaluate the score
        of each individual.
    :param individual: Array of model parameters: [t_rxn, V_des, a_des, d_des, d_lead, g_min]
    :param cf_collections: list of (Instance of Processed Data Class with vehicle trajectory data)
    :return: Individual Score = RMSE_dX: Float value indicating the individual score
    """

    t_rxn, V_f, a_f, d_des, d_lead, g_min = individual
    t_rxn = t_rxn / 10.

    # Calculate the number of points ahead that will be predicted (data collected at 10Hz):
    data_collection_rate = 0.1
    prediction_steps = t_rxn / data_collection_rate
    prediction_steps = int(prediction_steps)

    # Create lists of predicted and actual following vehicle velocity and vehicle separation distance:
    RMSE_list = list()

    for collection in cf_collections:  # Loop through each car-following collection (event) in the list
        v_f_pred_list = list()
        v_f_act_list = list()
        dX_pred_list = list()
        dX_act_list = list()
        vlead_act_list = list()
        alead_act_list = list()
        afoll_act_list = list()
        afoll_pred_list = list()

        # Initial variables considering reaction time
        for i in xrange(prediction_steps):
            # Initialize Variables for Clarity
            v_following_this = collection.v_foll[i]
            v_following_next = collection.v_foll[i + prediction_steps]
            v_lead_this = collection.v_lead[i]
            v_lead_next = collection.v_lead[i + prediction_steps]
            dX_this = collection.dX[i]
            dX_next = collection.dX[i + prediction_steps]

            # Predict Following Velocity
            v_f_pred_temp = gipps_predict_v_f(individual, v_following_this, v_lead_this, dX_this)
            v_f_pred_list.append(v_f_pred_temp)
            v_f_act_list.append(v_following_next)

            # Calculate New/Predicted Spacing
            acc_lead_temp = (v_lead_next - v_lead_this) / t_rxn
            dist_lead_temp = v_lead_this * t_rxn + 0.5 * acc_lead_temp * t_rxn ** 2
            acc_foll_temp = (v_f_pred_temp - v_following_this) / t_rxn
            dist_foll_temp = v_following_this * t_rxn + 0.5 * acc_foll_temp * t_rxn ** 2
            dX_pred_temp = dX_this - dist_foll_temp + dist_lead_temp

            dX_pred_list.append(dX_pred_temp)
            dX_act_list.append(dX_next)
            vlead_act_list.append(v_lead_next)
            alead_act_list.append(acc_lead_temp)
            afoll_act_list.append(acc_foll_temp)
            afoll_pred_list.append(np.nan)

        for i in xrange(len(collection.dX) - prediction_steps * 2):
            # Initialize Variables for Clarity
            v_following_this = v_f_pred_list[i]
            v_following_next = collection.v_foll[i + prediction_steps * 2]
            v_lead_this = collection.v_lead[i + prediction_steps]
            v_lead_next = collection.v_lead[i + prediction_steps * 2]  # times 2 because of first in previous loop
            dX_this = dX_pred_list[i]
            dX_next = collection.dX[i + prediction_steps * 2]

            # Predict Following Velocity
            v_f_pred_temp = gipps_predict_v_f(individual, v_following_this, v_lead_this, dX_this)

            # Calculate New/Predicted Spacing
            acc_lead_temp = (v_lead_next - v_lead_this) / t_rxn
            dist_lead_temp = v_lead_this * t_rxn + 0.5 * acc_lead_temp * t_rxn ** 2
            acc_foll_temp = (v_f_pred_temp - v_following_this) / t_rxn
            dist_foll_temp = v_following_this * t_rxn + 0.5 * acc_foll_temp * t_rxn ** 2
            dX_pred_temp = dX_this - dist_foll_temp + dist_lead_temp

            """
            # Crash Penalty
            if dX_pred_temp < 0:  # severe penalty if crash occurs.
                dX_pred_temp = np.inf
            """
            dX_pred_list.append(dX_pred_temp)
            dX_act_list.append(dX_next)
            v_f_pred_list.append(v_f_pred_temp)
            v_f_act_list.append(v_following_next)
            vlead_act_list.append(v_lead_next)
            alead_act_list.append(acc_lead_temp)
            afoll_act_list.append(collection.a_foll[i + prediction_steps * 2])
            afoll_pred_list.append(acc_foll_temp)

            del v_following_this, v_following_next, v_lead_this, v_lead_next, dX_this, dX_next
            del v_f_pred_temp, acc_lead_temp, dist_lead_temp, acc_foll_temp, dist_foll_temp, dX_pred_temp

        RMSE_dx = calc_RMSE(dX_pred_list, dX_act_list)
        RMSE_list.append(RMSE_dx)
        """
        f, (ax1, ax2, ax3) = plt.subplots(3)
        ax1.plot(dX_act_list, 'r')
        ax1.plot(dX_pred_list, 'b')
        ax1.set_title('Spacing')
        ax2.plot(v_f_act_list, 'r')
        ax2.plot(v_f_pred_list, 'b')
        ax2.plot(vlead_act_list, 'g')
        ax2.set_title('Velocity')
        ax3.plot(afoll_act_list, 'r', label='actual')
        ax3.plot(afoll_pred_list, 'b', label='predicted')
        ax3.plot(alead_act_list, 'g', label='lead')
        ax3.set_title('Acceleration')
        ax3.legend()
        f.subplots_adjust(hspace=0.5)
        f.suptitle('Gipps Sanity Check, Individual = {}, RMSE = {}'.format(individual, RMSE_dx))
        #plt.show()
        print RMSE_dx
        """

    # Take a weighted average of the RMSE from each CF event
    index = 0
    length_list = list()
    for collection in cf_collections:
        length_list.append(float(collection.point_count()))
        index += 1
    index = 0
    RMSE_list_weighted = list()
    for collection in cf_collections:
        factor = length_list[index] / np.sum(length_list)
        RMSE_list_weighted.append(RMSE_list[index] * factor)
        index += 1

    RMSE_all = np.sum(RMSE_list_weighted)
    # print RMSE_list



    # del v_f_pred_list, v_f_act_list, dX_pred_list, dX_act_list, t_rxn, V_f, a_f, d_des, d_lead, g_min

    if abs(d_des) > abs(d_lead):
        return RMSE_all,
        # return np.inf,
    else:
        return RMSE_all,


def gipps_predict_v_f(individual ,v_foll ,v_lead ,dX):
    """
    Predict the following vehicle's velocity
    :param individual: Array of model parameters: [t_rxn, V_des, a_des, d_des, d_lead, g_min]
    :param v_foll: Float of the following vehicle's velocity [m/s]
    :param v_lead: Float of the lead vehicle's velocity [m/s]
    :param dX: Float of the separation distance between the lead and following vehicle [m]
    :return: v_f_next_p: Predicted following velocity for next time stamp
    """

    t_rxn ,V_f ,a_f ,d_des ,d_lead ,g_min = individual
    t_rxn = t_rxn/ 10.  # sec
    V_f = V_f / 10.  # m/s
    a_f = a_f / 10.  # m/s2
    b_f = d_des / 10.  # m/s2
    b_l = d_lead / 10.  # m/s2
    g_min = g_min / 10.  # m

    # Solve for v_f_next
    try:
        v_f_acc_next = v_foll + 2.5 * a_f * t_rxn * (1 - v_foll / V_f) * sqrt(0.025 + v_foll / V_f)
    except ValueError:
        print individual
        print "{},{},{}".format(v_foll, v_lead, dX)
    try:
        v_f_dec_next = b_f * t_rxn + sqrt(
            b_f ** 2 * t_rxn ** 2 - (b_f * (2 * (dX - g_min) - v_foll * t_rxn - ((v_lead ** 2) / b_l))))
    except TypeError:
        v_f_dec_next = np.inf  # This occurs if no lead vehicle is present and will force the first equation to be used
    except ValueError:
        v_f_dec_next = 0  # This occurs with a negative square root in the equation for deceleration
        # print "ValueError: Negative Square Root"

    # ...we would decelerate at the maximum rate possible of the vehicle which we assume to be 0.6 m/s per tenth of
    # a second or a 6 m/s^2. This constant is taken from...
    # ...Gillespie (1992) in which he states that federal requirements say that a vehicle must
    # be able to stop from 60mph (26.8m/s) at an average rate of 6.1m/s^2
    # v_f_dec_next = max(v_f_dec_next,v_foll-0.6)  # This would need to be a function of the reaction time...
    # v_f_acc_next = min(v_f_acc_next,v_foll+0.6)
    # print "v || Accel: {} & Decel: {}".format(v_f_acc_next,v_f_dec_next)
    v_f_next_p = min(v_f_acc_next, v_f_dec_next)
    if v_f_next_p < 0:
        v_f_next_p = 0

    # del t_rxn,V_f,a_f,b_f,b_l,g_min,v_f_acc_next,v_f_dec_next,d_des,d_lead

    return v_f_next_p


def run_gipps_GA_old(cf_collections, cxpb, mutpb, m_indpb, ngen, npop, logfile, figfile=None):
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
    """
    t_rxn_min, t_rxn_max = 0.1, 3.0  # Adjusted from 0.4 to 0.1 because many results were right at 0.4 - Adjusted to integer
    V_des_min, V_des_max = 1.0, 40.0  # Adjusted from 29.6 to 40 and 10.4 to 1
    a_des_min, a_des_max = 0.8, 2.6
    d_des_min, d_des_max = -5.2, -1.6
    d_lead_min, d_lead_max = -4.5, -3.0
    g_min_min, g_min_max = 2.0, 5.0  # Increased from 4.0 when I realized my synthetic data was for 4.0

    t_rxn_min, t_rxn_max = 4, 30  # Adjusted from 0.4 to 0.1 because many results were right at 0.4 - Adjusted to integer
    V_des_min, V_des_max = 10, 400  # Adjusted from 29.6 to 40 and 10.4 to 1
    a_des_min, a_des_max = 8, 26
    d_des_min, d_des_max = -52, -16
    d_lead_min, d_lead_max = -52, -16
    g_min_min, g_min_max = 20, 50  # Increased from 4.0 when I realized my synthetic data was for 4.0
    """
    t_rxn_min, t_rxn_max = 4, 20
    V_des_min, V_des_max = 10, 400
    a_des_min, a_des_max = 8, 26
    d_des_min, d_des_max = -40, -10
    d_lead_min, d_lead_max = -40, -10
    g_min_min, g_min_max = 20, 50
    """
    t_rxn_min, t_rxn_max = 2, 30  # Adjusted from 0.4 to 0.1 because many results were right at 0.4 - Adjusted to integer
    V_des_min, V_des_max = 10, 400  # Adjusted from 29.6 to 40 and 10.4 to 1
    a_des_min, a_des_max = 8, 26
    d_des_min, d_des_max = -52, -16
    d_lead_min, d_lead_max = -45, -30
    g_min_min, g_min_max = 20, 100  # Increased from 4.0 when I realized my synthetic data was for 4.0

    # From Cuiffo, Punzo (2012) Calibration bible
    t_rxn_min, t_rxn_max = 2, 30
    V_des_min, V_des_max = 100, 400
    a_des_min, a_des_max = 1, 80
    d_des_min, d_des_max = -80, -1
    d_lead_min, d_lead_max = -80, -1
    g_min_min, g_min_max = 1, 100
    """
    toolbox.register("attr_t_rxn", random.randint, t_rxn_min, t_rxn_max)
    toolbox.register("attr_V_des", random.randint, V_des_min, V_des_max)
    toolbox.register("attr_a_des", random.randint, a_des_min, a_des_max)
    toolbox.register("attr_d_des", random.randint, d_des_min, d_des_max)
    toolbox.register("attr_d_lead", random.randint, d_lead_min, d_lead_max)
    toolbox.register("attr_g_min", random.randint, g_min_min, g_min_max)

    toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_t_rxn, toolbox.attr_V_des,
                                                                         toolbox.attr_a_des, toolbox.attr_d_des,
                                                                         toolbox.attr_d_lead, toolbox.attr_g_min), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # GA
    cx_indpb = 0.5  # percent of the individual that will be switched -- common to use 0.5 https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_and_half_uniform
    # m_indpb = 0.5  # probability of each attribute being swapped
    low = [t_rxn_min, V_des_min, a_des_min, d_des_min, d_lead_min, g_min_min]
    up = [t_rxn_max, V_des_max, a_des_max, d_des_max, d_lead_max, g_min_max]

    toolbox.register("mate", tools.cxUniform, indpb=cx_indpb)
    toolbox.register("mutate", tools.mutUniformInt, low=low, up=up, indpb=m_indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_gipps_GA, cf_collections=cf_collections)

    pop = toolbox.population(n=npop)

    log, best_score, best_indiv = evolve_gipps_GA(population=pop, toolbox=toolbox, cxpb=cxpb, mutpb=mutpb,
                                                  m_indpb=m_indpb, ngen=ngen, logfile=logfile)

    if figfile is not None:
        plt.plot(log['gen'], log['min_score'], label="{} {} {} {}".format(cxpb, mutpb, ngen, npop))
        plt.xlabel("Generation")
        plt.ylabel("Min Fitness of Population")
        plt.legend(loc="upper right")
        plt.savefig(figfile)

    return best_score, best_indiv



### Log Files ###
def initiate_gipps_calibration_log_file(file, cxpb, mutpb, m_indpb, pop_size, ngen):
    file.write('Gipps CFM Calibration - DEAP GA Implementation')
    file.write('\n')
    file.write('cxpb,mutpb,m_indpb,pop_size,ngen')
    file.write('\n')
    file.write('{},{},{},{},{}'.format(cxpb, mutpb, m_indpb, pop_size, ngen))
    file.write('\n')
    file.write('Gen,No Unique Indiv,Min Score,Ave Score,Max Score,Std Score,')
    file.write('Best: T_rxn,Best: V_des,Best: a_des,Best: d_des,Best: d_lead,Best: g_min')
    file.write('\n')

    print 'Gipps CFM Calibration - DEAP GA Implementation'
    print 'cxpb: {} | mutpb: {} | m_indpb: {} | pop_size: {} | ngen: {}'.format(cxpb, mutpb, m_indpb, pop_size, ngen)
    print '%4s | %4s | %8s | %8s | %8s | %8s | %4s, %4s, %4s, %4s, %4s, %4s' % (
    'gen', 'cnt', 'min', 'ave', 'max', 'std', 'T', 'V', 'a', 'd_f', 'd_l', 'g')


def append_to_gipps_calibration_log_file(file, gen, no_unique_indiv, min_score, ave_score, max_score, std_score,
                                              best_indiv):
    file.write('{},{},{},{},{},{},'.format(gen, no_unique_indiv, min_score, ave_score, max_score, std_score))
    file.write(
        '{},{},{},{},{},{}'.format(best_indiv[0] / 10., best_indiv[1] / 10., best_indiv[2] / 10., best_indiv[3] / 10.,
                                   best_indiv[4] / 10., best_indiv[5] / 10.))
    file.write('\n')

    print '%4.0f | %4.0f | %8.3f | %8.3f | %8.3f | %8.3f | %4.1f, %4.1f, %4.1f, %4.1f, %4.1f, %4.1f' % (
    gen, no_unique_indiv, min_score, ave_score, max_score, std_score, best_indiv[0] / 10., best_indiv[1] / 10.,
    best_indiv[2] / 10., best_indiv[3] / 10., best_indiv[4] / 10., best_indiv[5] / 10.)


def initiate_gipps_calibration_summary_file(file):
    file.write('iteration,time,cxpd,mutpd,m_indpb,ngen,npop,score,t_rxn,v_des,a_des,d_des,d_lead,g_min')
    file.write('\n')


def append_to_gipps_calibration_summary_file(file, elapsed_time, iteration, cxpb, mutpb, m_indpb, ngen, npop,
                                                  score, best_indiv):
    file.write('{},'.format(iteration))
    file.write('{},'.format(elapsed_time))
    file.write('{},{},{},{},{},'.format(cxpb, mutpb, m_indpb, ngen, npop))
    file.write('{},'.format(score))
    file.write(
        '{},{},{},{},{},{}'.format(best_indiv[0] / 10., best_indiv[1] / 10., best_indiv[2] / 10., best_indiv[3] / 10.,
                                   best_indiv[4] / 10., best_indiv[5] / 10.))
    file.write('\n')


def initiate_gipps_calibration_cs_summary_file(file):
    file.write('trip_set_no,trip_no,driver_id,adverse_cond,trip_cond,time,score,t_rxn,v_des,a_des,d_des,d_lead,g_min')
    file.write('\n')


def append_to_gipps_calibration_cs_summary_file(file, elapsed_time, trip_set_no, trip_no, driver_id,
                                                            adverse_cond, trip_cond, score, best_indiv):
    file.write('{},'.format(trip_set_no))
    file.write('{},'.format(trip_no))
    file.write('{},{},{},'.format(driver_id, adverse_cond, trip_cond))
    file.write('{},'.format(elapsed_time))
    file.write('{},'.format(score))
    file.write(
        '{},{},{},{},{},{}'.format(best_indiv[0] / 10., best_indiv[1] / 10., best_indiv[2] / 10., best_indiv[3] / 10.,
                                   best_indiv[4] / 10., best_indiv[5] / 10.))
    file.write('\n')

def initiate_201802Calib_gipps_summary_file(file):
    file.write('2018-02-15 Gipps Calibration Summary File')
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
    file.write('calibration_time_sec,calibration_score,t_rxn,v_des,a_des,d_des,d_lead,g_min')
    file.write('\n')


def append_to_201802Calib_gipps_summary_file(file, trip_no, driver_id, point_collection, cf_collections,
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
        '{},{},{},{},{},{}'.format(calib_best_indiv[0] / 10., calib_best_indiv[1] / 10., calib_best_indiv[2] / 10., calib_best_indiv[3] / 10.,
                                   calib_best_indiv[4] / 10., calib_best_indiv[5] / 10.))


    file.write("\n")


### Plotting/Analysis Functions ###
def gipps_sensitivity_analysis_plot(summary_file, date, save_path, CXPB, MUTPB, NGEN, NPOP):
    df = pd.read_csv(filepath_or_buffer=summary_file, delimiter=',', header=0)

    # Score Plot
    fig, axes = plt.subplots(nrows=len(CXPB), ncols=len(MUTPB), figsize=(15, 12))  # figsize=(13,11)
    fig.suptitle('Gipps Calibration Sensitivity Analysis | {} Generations | {}'.format(NGEN, date), fontsize=16,
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


def gipps_convergence_plot(log_file, date, save_path):
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


def gipps_sensitivity_analysis_file(summary_file, date, save_path, CXPB, MUTPB, M_INDPB, NGEN, NPOP):
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
