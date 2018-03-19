from __future__ import division
import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deap import base, creator, tools

"""
/*******************************************************************
FHWA CFM functions used to process and analyze CF Data. 

Author: Britton Hammit
E-mail: bhammit1@gmail.com
Date: 11-01-2017
********************************************************************/
"""

### FHWA Framework Binned Calibration Functions ###
def run_fhwa_fw_GA(binned_data, cxpb, mutpb, ngen, npop, logfile, figfile):
    """
        Main function for running the Gipps Genetic algorithm.
        :param binned_data: car-following event data are aggregated into bins using the "create_aggregate_plots" function
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

    toolbox.register("individual",tools.initIterate,creator.Individual,create_fhwa_fw_individual)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    # GA
    cx_indpb = 0.5  # percent of the individual that will be switched -- common to use 0.5 https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Uniform_and_half_uniform
    m_indpb = 0.5  # probability of each attribute being swapped

    toolbox.register("mate", cx_fhwa_fw_individuals, indpb=cx_indpb)
    toolbox.register("mutate", mutate_fhwa_fw_individual, indpb=m_indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_fhwa_fw_GA, binned_data=binned_data)

    pop = toolbox.population(n=npop)

    log, best_score, best_indiv = evolve_fhwa_fw_GA(population=pop, toolbox=toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                                                  logfile=logfile)

    plt.plot(log['gen'], log['min_score'], label="{} {} {} {}".format(cxpb, mutpb, ngen, npop))
    plt.xlabel("Generation")
    plt.ylabel("Min Fitness of Population")
    plt.legend(loc="upper right")
    plt.savefig(figfile)

    return best_score, best_indiv


def create_fhwa_fw_individual():
    """
    Creates a member of the population
    :return: Returns an "individual" in the format of an array.
    """

    test = False
    while test is False:
        # Static Variables
        G_cfmax = 800  # Set value - based on maximum distance considered in event extraction
        G_s = 50  # Set value - based on Algorithm Description Document Assignment of Gs for FW3

        # Dynamic/ Calibrated Variables
        v_a = random.randint(-150,0)
        v_s = random.randint(0,150)
        G_max = random.randint(1,G_cfmax)
        G_min = random.randint(1,G_max)
        G_c = random.randint(1,G_min)
        # Condition ensuring that the criteria of G_c > G_s holds true in initial assignment
        if G_c > G_s:
            test = True

    individual = [G_max,G_min,G_c,G_s,v_a,v_s]
    return individual


def mutate_fhwa_fw_individual(individual,indpb):

    G_max, G_min, G_c, G_s, v_a, v_s = individual

    for i in range(len(individual)):
        if random.random() < indpb:
            if i == 0:  # G_max
                individual[i] = random.randint(G_min,800)
                G_max = individual[i]
            elif i == 1:  # G_min
                individual[i] = random.randint(G_c,G_max)
                G_min = individual[i]
            elif i == 2:  # G_c
                G_c = individual[i]
                individual[i] = random.randint(G_s,G_min)
            elif i == 3:  # G_s - NOTHING
                pass
            elif i == 4:  # v_a
                individual[i] = random.randint(-150,0)
            elif i == 5:  # v_s
                individual[i] = random.randint(0,150)

    return individual,


def cx_fhwa_fw_individuals(ind1, ind2, indpb):
    """
    Modified from cxUniform
    Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals. The attributes are swapped accordingto the
    *indpb* probability.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    size = min(len(ind1), len(ind2))

    for i in xrange(size):
        if random.random() < indpb:
            if i == 0:  # G_max
                if ind1[i] > ind2[i+1]:  # > G_min
                    ind2[i] = ind1[i]
                if ind2[i] > ind1[i+1]:
                    ind1[i] = ind2[i]

            elif i == 1:  # G_min
                if ind1[i] < ind2[i-1] and ind1[i] > ind2[i+1]:  # < G_max and > G_c
                    ind2[i] = ind1[i]
                if ind2[i] < ind1[i-1] and ind2[i] > ind1[i+1]:
                    ind1[i] = ind2[i]

            elif i == 2:  # G_c
                if ind1[i] < ind2[i-1]:
                    ind2[i] = ind1[i]
                if ind2[i] < ind1[i-1]:
                    ind1[i] = ind2[i]

            elif i == 3:  # G_s
                pass

            elif i == 4:  # v_a
                ind1[i], ind2[i] = ind2[i], ind1[i]

            elif i == 5:  # v_s
                ind1[i], ind2[i] = ind2[i], ind1[i]

    return ind1, ind2


def evolve_fhwa_fw_GA(population, toolbox, cxpb, mutpb, ngen, logfile):
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

    initiate_deap_fhwa_fw_calibration_log_file(file=logfile, cxpb=cxpb, mutpb=mutpb, pop_size=len(population), ngen=ngen)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof_global = tools.HallOfFame(1)
    hof_global.update(population)

    record = stats.compile(population) if stats else {}

    append_to_deap_fhwa_fw_calibration_log_file(file=logfile, gen=0,
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

        append_to_deap_fhwa_fw_calibration_log_file(file=logfile, gen=gen,
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

    append_to_deap_fhwa_fw_calibration_log_file(file=logfile, gen=gen, no_unique_indiv=len(invalid_ind),
                                              min_score=best_score, ave_score=0, max_score=0,
                                              std_score=0, best_indiv=best_indiv)

    return log, best_score, best_indiv


def evaluate_fhwa_fw_GA(individual,binned_data):
    """
    Generate the score for an individual using two "score levels" associated with the fhwa car following
    model framework:
    1. Regime Score = standard deviation (acceleration points) within a regime
    2. Individual Score = ___________ (regime scores) within an individual framework
    :param individual: Array of model parameters: [zero,G_s,G_c,G_min,G_max,G_cfmax,v_a,v_s]
    :param binned_data:
    :param log_file: (optional) CSV File variable with filename.csv and save path
    = open(os.path.join(filename.csv,path),'w') & If included will provide descriptive information about
    each individual's parameter configuration.
    :return: Float value indicating the individual score
    """
    G_max, G_min, G_c, G_s, v_a, v_s = individual
    G_max = G_max/10.
    G_min = G_min/10.
    G_c = G_c/10.
    G_s = G_s/10.
    v_a = v_a/10.
    v_s = v_s/10.

    # Cluster points into regimes based on location on dV-dX plot:
    # ----------------------------------------------------------------------------------------
    # Create empty lists to aggregate the acceleration values of each point in each regime
    no_rxn_accel = list(); regime1_accel = list(); regime2_accel = list(); regime3_accel = list()
    regime4_accel = list(); regime5_accel = list(); regime6_accel = list(); regime7_accel = list()

    # Iterate through data points and assign points to the appropriate regime according to the
    # framework configuration:
    for i in range(len(binned_data) - 1):
        dX = binned_data[i+1][0]
        dX_index = i + 1
        # Assign No Reaction Zone
        if dX >= G_max:
            for j in range(len(binned_data[dX_index]) - 1):
                dV_index = j + 1
                no_rxn_accel.append(binned_data[dX_index][dV_index])
        elif dX >= G_min:
            for j in range(len(binned_data[dX_index]) - 1):
                dV = binned_data[0][j]
                dV_index = j + 1
                if dV < v_a:
                    regime1_accel.append(binned_data[dX_index][dV_index])

                elif dV >= v_a and dV < v_s:
                    no_rxn_accel.append(binned_data[dX_index][dV_index])

                elif dV >= v_s:
                    regime2_accel.append(binned_data[dX_index][dV_index])
        elif dX >= G_c:
            for j in range(len(binned_data[dX_index]) - 1):
                dV = binned_data[0][j]
                dV_index = j + 1
                if dV < v_a:
                    regime3_accel.append(binned_data[dX_index][dV_index])

                elif dV >= v_a and dV < v_s:
                    regime4_accel.append(binned_data[dX_index][dV_index])

                elif dV >= v_s:
                    regime5_accel.append(binned_data[dX_index][dV_index])
        elif dX >= G_s:
            for j in range(len(binned_data[dX_index]) - 1):
                dV_index = j + 1
                regime6_accel.append(binned_data[dX_index][dV_index])
        else:
            for j in range(len(binned_data[dX_index]) - 1):
                dV_index = j + 1
                regime7_accel.append(binned_data[dX_index][dV_index])

    # Aggregate lists of accelerations for computation simplicity:
    regime_accelerations = [no_rxn_accel, regime1_accel, regime2_accel, regime3_accel, regime4_accel, regime5_accel,
                            regime6_accel, regime7_accel]

    # Calculate the score for each regime:
    # ----------------------------------------------------------------------------------------
    # The penalty value will be assigned as the regime score for any regime that does not have any points.
    penalty = np.inf
    regime_score_list = [np.nan for i in range(len(regime_accelerations))]
    for i in range(len(regime_score_list)):
        if np.isnan(np.nanmean(regime_accelerations[i])):
            regime_score_list[i] = penalty
        else:

            # Squared Error
            sum = 0
            mean = np.nanmean(regime_accelerations[i])
            for j in range(len(regime_accelerations[i])):
                if np.isnan(regime_accelerations[i][j]):
                    pass
                else:
                    sum += (regime_accelerations[i][j] - mean) ** 2
            regime_score_list[i] = sum
            """
            # Standard Deviation
            regime_score_list[i] = np.nanstd(regime_accelerations[i])
            """
    no_rxn_score, regime1_score, regime2_score, regime3_score, regime4_score, regime5_score, regime6_score, regime7_score = regime_score_list

    # Eliminated Regime 7 because Gs is now a fixed value.
    regime_scores_used = [no_rxn_score, regime1_score, regime2_score, regime3_score, regime4_score, regime5_score, regime6_score]

    # Calculate the score for each individual:
    # ----------------------------------------------------------------------------------------
    # The penalty value will be assigned as the regime score for any regime that does not have any points.
    if np.isinf(np.nanmean(regime_scores_used)):
        individual_score = np.inf
    else:
        # Change Individual Score Here!
        individual_score = np.nanmean(regime_scores_used)

    # Record the logic for the log file:
    logic = "Regime Score = SE of Accelerations && Summary Score = Sum - {} penalty for no " \
            "points in a regime".format(penalty)

    return individual_score,


### FHWA Framework Calibration Summary/Log File/Plotting Functions ###
def initiate_deap_fhwa_fw_calibration_log_file(file,cxpb,mutpb,pop_size,ngen):

    file.write('FHWA CFM Framework Calibration - DEAP GA Implementation')
    file.write('\n')
    file.write('cxpb,mutpb,pop_size,ngen')
    file.write('\n')
    file.write('{},{},{},{}'.format(cxpb,mutpb,pop_size,ngen))
    file.write('\n')
    file.write('Gen,No Unique Indiv,Min Score,Ave Score,Max Score,Std Score,')
    file.write('Best: G_max,Best: G_min,Best: G_c,Best: G_s,Best: v_a,Best: v_s')

    file.write('\n')

    print 'FHWA CFM Framework Calibration - DEAP GA Implementation'
    print 'cxpb: {} | mutpb {} | pop_size: {} | ngen: {}'.format(cxpb,mutpb,pop_size,ngen)
    print '%4s | %4s | %8s | %8s | %8s | %8s | %4s, %4s, %4s, %4s, %4s, %4s' %('gen','cnt','min','ave','max','std','G_max','G_min','G_c','G_s','v_a','v_s')


def append_to_deap_fhwa_fw_calibration_log_file(file,gen,no_unique_indiv,min_score,ave_score,max_score,std_score,best_indiv):

    file.write('{},{},{},{},{},{},'.format(gen,no_unique_indiv,min_score,ave_score,max_score,std_score))
    file.write('{},{},{},{},{},{}'.format(best_indiv[0]/10.,best_indiv[1]/10.,best_indiv[2]/10.,best_indiv[3]/10.,best_indiv[4]/10.,best_indiv[5]/10.))
    file.write('\n')

    print '%4.0f | %4.0f | %8.3f | %8.3f | %8.3f | %8.3f | %4.1f, %4.1f, %4.1f, %4.1f, %4.1f, %4.1f' %(gen,no_unique_indiv,min_score,ave_score,max_score,std_score,best_indiv[0]/10.,best_indiv[1]/10.,best_indiv[2]/10.,best_indiv[3]/10.,best_indiv[4]/10.,best_indiv[5]/10.)


def initiate_deap_fhwa_fw_calibration_summary_file(file):
    file.write('iteration,time,cxpd,mutpd,ngen,npop,score,G_max,G_min,G_c,G_s,v_a,v_s')
    file.write('\n')


def append_to_deap_fhwa_fw_calibration_summary_file(file,elapsed_time,iteration,cxpb,mutpb,ngen,npop,score,best_indiv):
    file.write('{},'.format(iteration))
    file.write('{},'.format(elapsed_time))
    file.write('{},{},{},{},'.format(cxpb,mutpb,ngen,npop))
    file.write('{},'.format(score))
    file.write('{},{},{},{},{},{}'.format(best_indiv[0]/10.,best_indiv[1]/10.,best_indiv[2]/10.,best_indiv[3]/10.,best_indiv[4]/10.,best_indiv[5]/10.))
    file.write('\n')


def deap_fhwa_fw_sensitivity_analysis_plot(summary_file,date,save_path,CXPB,MUTPB,NGEN,NPOP):

    df = pd.read_csv(filepath_or_buffer=summary_file,delimiter=',',header=0)

    # Score Plot
    fig, axes = plt.subplots(nrows=len(CXPB),ncols=len(MUTPB),figsize=(13,11))
    fig.suptitle('FHWA CFM Framework Calibration Sensitivity Analysis | {} Generations | {}'.format(NGEN,date), fontsize=16, fontweight='bold')
    for i in range(len(CXPB)):
        for j in range(len(MUTPB)):
            # Create data frame for specific plot
            df_temp = df[(df.cxpd == CXPB[i]) & (df.mutpd == MUTPB[j])]
            no_iterations = len(df_temp[df_temp.npop == NPOP[0]])
            df_temp.plot(x='npop',y='score',kind='scatter',subplots=True,ax=axes[i,j],label='{} Iterations'.format(no_iterations),color='b')
            axes[i,j].set_title('cxpb: {} | mutpb: {}'.format(CXPB[i],MUTPB[j]),fontweight='bold')

            # Y Limits
            axes[i,j].set_ylim([0.1,0.35])
            axes[i,j].set_ylabel('Score: RMSE of dX [m]',fontsize=12)

            # X Limits
            min_pop = min(NPOP)
            max_pop = max(NPOP)
            diff_pop = max_pop - min_pop
            buffer_dist = diff_pop*0.2/0.6  # 20% buffer on each side
            axes[i,j].set_xlim([min_pop-buffer_dist,max_pop+buffer_dist])
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
            x_buffer_dist = diff_pop*0.15/0.7  # 10% buffer on each side
            y_buffer_dist = (0.35-0.1)*0.02/0.96  # 2% buffer above
            bbox = dict(boxstyle="round,pad=0.1", fc='white', ec='white', lw=0, alpha=0.8)
            for k in range(len(NPOP)):
                axes[i,j].plot((NPOP[k]-x_buffer_dist,NPOP[k]+x_buffer_dist),(ave_list[k],ave_list[k]),color='r',linestyle='--',label='ave')
                axes[i,j].annotate(('Ave: {:4.3f}'.format(ave_list[k])),xy=(NPOP[k]-x_buffer_dist,ave_list[k]+y_buffer_dist),xytext=(NPOP[k]-x_buffer_dist,ave_list[k]+y_buffer_dist),color='g',bbox=bbox,fontweight='bold')
                axes[i,j].annotate(('Std: {:4.3f}'.format(std_list[k])),xy=(NPOP[k]-x_buffer_dist,ave_list[k]-y_buffer_dist*2.5),xytext=(NPOP[k]-x_buffer_dist,ave_list[k]-y_buffer_dist*2.5),color='g',bbox=bbox,fontweight='bold')

            del min_pop,max_pop,diff_pop,npop,ave_list,std_list,k,x_buffer_dist,y_buffer_dist,bbox

            del df_temp

    plt.subplots_adjust(hspace=0.35, wspace=0.4)
    #plt.show()

    fig.savefig(os.path.join(save_path,'{}'.format(date)))

