from __future__ import division
import time
import numpy as np
from classes import VissimSyntheticDataPoint
from classes import VissimSyntheticCollection
from classes import NewList

"""
/*******************************************************************
Synthetic Data functions used to process and analyze CF Data. 

Author: Britton Hammit
E-mail: bhammit1@gmail.com
Date: 12-01-2017
********************************************************************/
"""

### Synthetic Data ###
def import_vissim_synthetic_data(filename,path):
    file = open(os.path.join(path, filename), 'r')  # Open Vehicle Data file
    for i in range(1):  # Number of header lines
        file.next()  # Header Line
    data = []
    for line in file:
        data.append(line.strip().split(','))
    file.close()

    # Convert data from Float to Integer
    for i in range(len(data)):
        for j in range(len(data[i])):
            try:
                data[i][j] = float(data[i][j])
            except ValueError:
                data[i][j] = np.nan

    # Convert into SyntheticDataPoints
    data_points = list()
    for i in range(len(data)):
        data_points.append(VissimSyntheticDataPoint(data[i]))

    # Convert to Point Collection
    return VissimSyntheticCollection(data_points)


def initiate_synthetic_trip_cf_summary_file(file):
    file.write('Trip Car-Following Summary File: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')
    file.write('Veh No,Start Time,Stop Time,')

    file.write('% Time in CF,Time in CF [min],No. CF Events,')

    stats_operations_names = ['Mean', 'Max', 'Min', 'Stdev']
    measure_names = ['v_foll [m/s]', 'a_foll [m/s2]', 'dX [m]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    stats_operations_names = ['Min', 'Mean', 'Percentile15']
    measure_names = ['time to collision [s]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            if i == len(stats_operations_names) - 1 and j == len(measure_names) - 1:
                file.write(stats_operations_names[i] + ': ' + measure_names[j])
            else:
                file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    file.write("\n")


def append_to_synthetic_trip_cf_summary_file(point_collection,cf_collections,file):
    file.write('{},{},{},'.format(point_collection[0].vehicle_no,point_collection[0].timestamp,point_collection[-1].timestamp))

    file.write("{},{},{},".format(point_collection.percent_car_following(), point_collection.time_car_following(),
                                        len(cf_collections)))

    operations_names = ['mean', 'maximum', 'minimum', 'standard_deviation']
    measure_names = ['v_foll', 'a_foll', 'dX']

    measure_values = []
    for j in range(len(measure_names)):
        temp = list()
        for k in range(point_collection.point_count()):
            temp.append(getattr(point_collection.list_of_data_points[k], measure_names[j]))
        measure_values.append(NewList(temp))
    del temp

    for j in range(len(operations_names)):
        for k in range(len(measure_names)):
            temp = getattr(measure_values[k], operations_names[j])
            file.write("{},".format(temp()))

    measure_values = NewList(point_collection.time_to_collision())
    file.write("{},{},{}".format(measure_values.minimum(), measure_values.mean(), measure_values.percentile(15)))

    file.write("\n")


def initiate_synthetic_cfe_summary_file(file):
    file.write('Car-Following Event Segmentation Summary File: {}'.format(
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')
    file.write('Veh No,CFE No,')
    file.write('Distance Traveled [km],Duration [sec],')
    file.write('Start Time,Stop Time,')

    stats_operations_names = ['Mean', 'Max', 'Min', 'Median', 'Percentile85', 'Stdev', 'Variance', 'Coeff of Variation']
    measure_names = ['v_foll [m/s]', 'a_foll [m/s2]', 'dX [m]', 'v_lead [m/s]', 'a_lead [m/s2]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    stats_operations_names = ['Min', 'Mean', 'Percentile15']
    measure_names = ['time to collision [s]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            if i == len(stats_operations_names) - 1 and j == len(measure_names) - 1:
                file.write(stats_operations_names[i] + ': ' + measure_names[j])
            else:
                file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    file.write("\n")


def append_to_synthetic_cfe_summary_file(point_collection,cf_collections,file):

    if len(cf_collections)==0:
        file.write("{},".format(point_collection[0].vehicle_no))
        file.write("\n")

    for i in range(len(cf_collections)):
        file.write("{},".format(point_collection[0].vehicle_no))
        file.write("{},".format(i+1))
        file.write("{},{},".format(cf_collections[i].dist_traveled(),cf_collections[i].time_elapsed()))

        start_time, stop_time = cf_collections[i].start_stop_timestamp()
        file.write("{},{},".format(start_time, stop_time))

        measure_names = ['v_foll', 'a_foll', 'dX', 'v_lead', 'a_lead']
        measure_values = []
        for j in range(len(measure_names)):
            temp = list()
            for k in range(cf_collections[i].point_count()):
                temp.append(getattr(cf_collections[i].list_of_data_points[k], measure_names[j]))
            measure_values.append(NewList(temp))
        del temp
        operations_names = ['mean', 'maximum', 'minimum', 'median', 'percentile', 'standard_deviation', 'variance',
                            'coeff_variation']
        for j in range(len(operations_names)):
            for k in range(len(measure_names)):
                temp = getattr(measure_values[k], operations_names[j])
                file.write("{},".format(temp()))
        del temp

        measure_values = NewList(cf_collections[i].time_to_collision())
        file.write("{},{},{}".format(measure_values.minimum(), measure_values.mean(), measure_values.percentile(15)))

        file.write("\n")