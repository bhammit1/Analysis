from __future__ import division
import numpy as np

"""
/*******************************************************************
Bin Aggregation functions used to process and analyze CF Data. 

Author: Britton Hammit
E-mail: bhammit1@gmail.com
Date: 08-01-2017
********************************************************************/
"""

### FHWA Framework Binning Functions ###
def create_aggregate_plots_old(plot_content,plot_evaluation,point_collection,file = None):
    """
    Function creates the aggregated binned plots that can be used for evaluation of car-following data
    from multiple drivers.
    :param plot_content: the vehicle data values to be used in the plot - "acceleration, velocity"
    :param plot_evaluation: the metric by which the plots will be generated - "mean, stdev, frequency"
    :param vehicle_data:
    :param file: output file [.csv] to save binned data - Not Required
    :return: Saves .csv file with aggregated plots
    """

    # Print csv file with average acceleration values for viewing
    matrix_bounds_dV = np.arange(-9.75,10.25,0.5)
    matrix_bounds_dX = np.arange(75.5,0.5,-1)

    data_matrix = [[list() for i in range(len(matrix_bounds_dV)-1)] for j in
                       range(len(matrix_bounds_dX)-1)]
    processed_data_matrix = [[np.nan for i in range(len(matrix_bounds_dV))] for j in
                       range(len(matrix_bounds_dX))]

    if plot_content == 'acceleration':
        for i in range(len(point_collection.a_foll)):
            for j in range(len(data_matrix)):  #dX
                if point_collection.dX[i] < matrix_bounds_dX[j] and point_collection.dX[i] >= matrix_bounds_dX[j+1]:
                    for k in range(len(data_matrix[0])):  #dV
                        if point_collection.dV[i] > matrix_bounds_dV[k] and point_collection.dV[i] <= matrix_bounds_dV[k+1]:
                            data_matrix[j][k].append(point_collection.a_foll[i])
                            break

    elif plot_content == 'velocity':
        for i in range(len(point_collection.v_foll)):
            for j in range(len(data_matrix)):
                if point_collection.dX[i] < matrix_bounds_dX[j] and point_collection.dX[i] >= matrix_bounds_dX[j+1]:
                    for k in range(len(data_matrix[0])):
                        if point_collection.dV[i] > matrix_bounds_dV[k] and point_collection.dV[i] <= matrix_bounds_dV[k+1]:
                            data_matrix[j][k].append(point_collection.v_foll[i])
                            break

    else:
        data_matrix = None

    if plot_evaluation == 'mean':
        # This is where we can update the output to be exactly what andy's is!
        #   Minimum of 2 points && Variance under 1m/s2
        for i in range(len(data_matrix)):
            for j in range(len(data_matrix[0])):
                if len(data_matrix[i][j]) > 2:  # Requirement from the frequency of points
                    if np.nanstd(data_matrix[i][j]) < np.sqrt(1):  # Requirement from the stdev of the points
                        processed_data_matrix[i+1][j+1] = np.nanmean(data_matrix[i][j])
            processed_data_matrix[i+1][0] = matrix_bounds_dX[i+1]
        for i in range(len(processed_data_matrix[0])):
            processed_data_matrix[0][i] = matrix_bounds_dV[i]

    else:
        processed_data_matrix = None

    # If Target file provided, print to file.
    if file is not None:
        file.write('\n')
        for i in range(len(processed_data_matrix)):
            file.write('{},'.format(matrix_bounds_dX[i]))
            for j in range(len(processed_data_matrix[0])):
                if j != len(processed_data_matrix[0]) - 1:
                    file.write('{},'.format(processed_data_matrix[i][j]))
                else:
                    file.write('{}'.format(processed_data_matrix[i][j]))
            file.write('\n')
        file.write('dX/dV,')
        for i in range(len(matrix_bounds_dV)):
            if i != len(matrix_bounds_dV) - 1:
                file.write('{},'.format(matrix_bounds_dV[i]))
            else:
                file.write('{}'.format(matrix_bounds_dV[i]))
        file.close()

    return processed_data_matrix


def reaggregate_bins(binned_data):
    """
    This was used for the TRB paper - T-test Analysis.
    :param binned_data:
    :return:
    """
    row_index = np.arange(4,76+6,6)  # 6meters for car-length. Starting at 4 for even bins, cutting from top
    col_index = np.arange(2,37+5,5)  # 2.5 m/s = 5 bins, which is approx 5mph. Starting at 2 and ending before 39 for even bins, cutting two from each side.

    list_bin = [[list() for i in range(len(col_index)-1)] for j in range(len(row_index)-1)]

    for i in range(len(binned_data)):  #row
        for j in range(len(binned_data[i])):  #col
            for k in range(len(row_index)-1):
                for m in range(len(col_index)-1):
                    if i>row_index[k] and i<=row_index[k+1]:
                        if j>col_index[m] and j<=col_index[m+1]:
                            list_bin[k][m].append(binned_data[i][j])

    return list_bin


def create_aggregate_plots(plot_content,plot_evaluation,cf_collections,file = None):
    """
    Function creates the aggregated binned plots that can be used for evaluation of car-following data
    from multiple drivers.
    :param plot_content: the vehicle data values to be used in the plot - "acceleration, velocity"
    :param plot_evaluation: the metric by which the plots will be generated - "mean, stdev, frequency"
    :param vehicle_data:
    :param file: output file [.csv] to save binned data - Not Required
    :return: Saves .csv file with aggregated plots
    """

    # Print csv file with average acceleration values for viewing
    matrix_bounds_dV = np.arange(-9.75,10.25,0.5)
    matrix_bounds_dX = np.arange(75.5,-1.5,-1)

    data_matrix = [[list() for i in range(len(matrix_bounds_dV)-1)] for j in
                       range(len(matrix_bounds_dX)-1)]
    processed_data_matrix = [[np.nan for i in range(len(matrix_bounds_dV))] for j in
                       range(len(matrix_bounds_dX))]

    for collection in cf_collections:

        if plot_content == 'acceleration':
            for i in range(len(collection.a_foll)):
                for j in range(len(data_matrix)):  #dX
                    if collection.dX[i] < matrix_bounds_dX[j] and collection.dX[i] >= matrix_bounds_dX[j+1]:
                        for k in range(len(data_matrix[0])):  #dV
                            if collection.dV[i] > matrix_bounds_dV[k] and collection.dV[i] <= matrix_bounds_dV[k+1]:
                                data_matrix[j][k].append(collection.a_foll[i])
                                break

        elif plot_content == 'velocity':
            for i in range(len(collection.v_foll)):
                for j in range(len(data_matrix)):
                    if collection.dX[i] < matrix_bounds_dX[j] and collection.dX[i] >= matrix_bounds_dX[j+1]:
                        for k in range(len(data_matrix[0])):
                            if collection.dV[i] > matrix_bounds_dV[k] and collection.dV[i] <= matrix_bounds_dV[k+1]:
                                data_matrix[j][k].append(collection.v_foll[i])
                                break

        else:
            data_matrix = None

    if plot_evaluation == 'mean':
        # This is where we can update the output to be exactly what andy's is!
        #   Minimum of 2 points && Variance under 1m/s2
        for i in range(len(data_matrix)):
            for j in range(len(data_matrix[0])):
                if len(data_matrix[i][j]) > 2:  # Requirement from the frequency of points
                    if np.nanstd(data_matrix[i][j]) < np.sqrt(1):  # Requirement from the stdev of the points
                        processed_data_matrix[i+1][j+1] = np.nanmean(data_matrix[i][j])
            processed_data_matrix[i+1][0] = matrix_bounds_dX[i+1]
        for i in range(len(processed_data_matrix[0])):
            processed_data_matrix[0][i] = matrix_bounds_dV[i]

    else:
        processed_data_matrix = None

    # If Target file provided, print to file.
    if file is not None:
        file.write('\n')
        for i in range(len(processed_data_matrix)):
            file.write('{},'.format(matrix_bounds_dX[i]))
            for j in range(len(processed_data_matrix[0])):
                if j != len(processed_data_matrix[0]) - 1:
                    file.write('{},'.format(processed_data_matrix[i][j]))
                else:
                    file.write('{}'.format(processed_data_matrix[i][j]))
            file.write('\n')
        file.write('dX/dV,')
        for i in range(len(matrix_bounds_dV)):
            if i != len(matrix_bounds_dV) - 1:
                file.write('{},'.format(matrix_bounds_dV[i]))
            else:
                file.write('{}'.format(matrix_bounds_dV[i]))
        file.close()

    return processed_data_matrix