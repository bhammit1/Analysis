from __future__ import division
import time
import warnings
import numpy as np
from classes import NewList

"""
/*******************************************************************
Wy NDS Data Analysis Tool functions used to process and analyze CF Data. 

Author: Britton Hammit
E-mail: bhammit1@gmail.com
Date: 10-04-2017
********************************************************************/
"""

## NDS Summary File Initialization Functions ##
def initiate_ts_summary_file(file):
    file.write('Time Segmentation Summary File: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')
    file.write('EventID_Segment,EventID,Segment,Segment Length(min),Distance Traveled [km],')
    file.write('Time Bin,Day,Month,Year,')
    file.write('Start Time,Stop Time,Start Lat,Stop Lat,Start Long,Stop Long,')
    file.write('Is Freeway?,Weather Condition,Surface Condition,Visibility,Traffic Condition,')
    file.write('Wiper: nan,Wiper: 0.0,Wiper: 1.0,Wiper: 2.0,Wiper: 3.0,Wiper: 254,Wiper: 255,')

    stats_operations_names = ['Mean', 'Max', 'Min', 'Median', 'Percentile85', 'Stdev', 'Variance', 'Coeff of Variation']
    measure_names = ['v_foll [m/s]', 'a_foll [m/s2]', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway [s]', 'dX [m]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    file.write('Percent Time Car-Following,Max Continuous Car-Following [s],Number of Lead Vehicles,Min Time to Collision [s]')

    file.write("\n")


def initiate_cfs_summary_file(file):
    file.write('Car-Following Event Segmentation Summary File: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')
    file.write('Trip EventID_CF EventID,EventID,CF EventID,Distance Traveled [km],')
    file.write('Duration [sec],')
    file.write('STAC Availability,')
    file.write('Time Bin,Day,Month,Year,')
    file.write('Start Time,Stop Time,Start Lat,Stop Lat,Start Long,Stop Long,')
    file.write('Is Freeway?,Weather Condition,Surface Condition,Visibility,Traffic Condition,')
    file.write('Wiper: nan,Wiper: 0.0,Wiper: 1.0,Wiper: 2.0,Wiper: 3.0,Wiper: 254,Wiper: 255,')

    stats_operations_names = ['Mean', 'Max', 'Min', 'Median', 'Percentile85', 'Stdev', 'Variance', 'Coeff of Variation']
    measure_names = ['v_foll [m/s]', 'a_foll [m/s2]', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway [s]', 'dX [m]', 'dV [m/s]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    stats_operations_names = ['Min','Mean','Percentile15','Median']
    measure_names = ['time to collision [s]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            if i == len(stats_operations_names)-1 and j == len(measure_names)-1:
                file.write(stats_operations_names[i] + ': ' + measure_names[j])
            else:
                file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    file.write("\n")


def initiate_var_exist_summary_file(file,point_collection):
    file.write('Variable Existence Summary File: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')
    file.write("Trip Number,")
    variable_list = point_collection[0].__dict__.keys()
    variable_list.sort()
    variable_list.remove('data')
    variable_list.remove('index')
    variable_list.remove('stac_data_available')
    variable_list.remove('dV')
    counter=0
    for variable in variable_list:
        if counter != len(variable_list)-1:
            file.write(variable + ",")
        else:
            file.write(variable)
        counter+=1
    file.write("\n")


def initiate_trip_cf_stats_summary_file(file):
    file.write('Trip Car-Following Summary File: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')
    file.write('Summary statistics describing contained car-following events - note that these statistics only describe the CF events NOT all time series data')
    file.write('\n')
    file.write('Trip Number,Driver ID,Trip Length [min],Distance Traveled [km],')
    file.write('STAC Availability,')
    file.write('Time Bin,Day,Month,Year,')
    file.write('Start Time,Stop Time,Start Lat,Stop Lat,Start Long,Stop Long,')

    file.write('% Time in CF,Time in CF [min],No. CF Events,Time CF Events [min],No. Targets Identified,No. Lead Targets Identified,')

    stats_operations_names = ['Mean', 'Max', 'Min', 'Median', 'Stdev', 'Variance', 'Percentile85']
    measure_names = ['v_foll [m/s]','a_foll [m/s2]','headway [s]','dX [m]','dV [m/s]','v_lead [m/s]','a_lead [m/s2]','time to collision [s]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    file.write("\n")


def initiate_trip_enviro_stats_summary_file(file):
    file.write('Trip Enviro & Stats Summary File: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')
    file.write('Trip Information,')
    file.write(',,,,,,,,,,,,')
    file.write('Freeway,,Weather,,,,,,,Surface,,,,Visibility,,,Traffic,,,,,,Wiper Status,,,,,,,Statistics')
    file.write('\n')

    file.write('Trip Number,Trip Length [min],Distance Traveled [km],')
    file.write('Time Bin,Day,Month,Year,')
    file.write('Start Time,Stop Time,Start Lat,Stop Lat,Start Long,Stop Long,')
    file.write('0-Not Freeway,1-Freeway,')
    file.write('1-Clear,2-Light Rain,3-Heavy Rain,4-Snowing,5-Fog,6-Sleeting,7-Mist/Light Rain,')
    file.write('1-Dry,2-Wet,3-Snowy,4-Icy,')
    file.write('1,2,3,')
    file.write('1-LOS A,2-LOS B,3-LOS C,4-LOS D,5-LOS E,6-LOS F,')
    file.write('Wiper: nan,Wiper: 0.0,Wiper: 1.0,Wiper: 2.0,Wiper: 3.0,Wiper: 254,Wiper: 255,')

    stats_operations_names = ['Mean', 'Max', 'Min', 'Median', 'Percentile85', 'Stdev', 'Variance', 'Coeff of Variation']
    measure_names = ['v_foll [m/s]', 'a_foll [m/s2]', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway [s]', 'dX [m]', 'vtti_accel_y',
                         'vtti_light_level', 'vtti_head_confidence',
                         'vtti_head_position_x','vtti_head_position_x_baseline',
                         'vtti_head_position_y', 'vtti_head_position_y_baseline',
                         'vtti_head_position_z', 'vtti_head_position_z_baseline',
                         'vtti_head_rotation_x', 'vtti_head_rotation_x_baseline',
                         'vtti_head_rotation_y', 'vtti_head_rotation_y_baseline',
                         'vtti_head_rotation_z', 'vtti_head_rotation_z_baseline']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    stats_operations_names = ['Min','Mean','Percentile15']
    measure_names = ['time to collision [s]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            if i == len(stats_operations_names)-1 and j == len(measure_names)-1:
                file.write(stats_operations_names[i] + ': ' + measure_names[j])
            else:
                file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    file.write('\n')


def initiate_trip_cf_enviro_stats_summary_file(file):
    file.write('Trip Car-Following & Enviro & Stats Summary File: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')

    file.write('Trip Information,')
    file.write(',,,,,,,,,,,,,,,,,,')
    file.write('Freeway,,Weather,,,,,,,Surface,,,,Visibility,,,Traffic,,,,,,Wiper Status,,,,,,,Statistics')
    file.write('\n')

    file.write('Trip Number,Trip Length [min],Distance Traveled [km],')
    file.write('STAC Availability,')
    file.write('Time Bin,Day,Month,Year,')
    file.write('Start Time,Stop Time,Start Lat,Stop Lat,Start Long,Stop Long,')

    file.write('% Time in CF,Time in CF [min],No. CF Events,No. Targets Identified,No. Lead Targets Identified,')

    file.write('0-Not Freeway,1-Freeway,')
    file.write('1-Clear,2-Light Rain,3-Heavy Rain,4-Snowing,5-Fog,6-Sleeting,7-Mist/Light Rain,')
    file.write('1-Dry,2-Wet,3-Snowy,4-Icy,')
    file.write('1,2,3,')
    file.write('1-LOS A,2-LOS B,3-LOS C,4-LOS D,5-LOS E,6-LOS F,')
    file.write('Wiper: nan,Wiper: 0.0,Wiper: 1.0,Wiper: 2.0,Wiper: 3.0,Wiper: 254,Wiper: 255,')

    stats_operations_names = ['Mean', 'Max', 'Min', 'Median', 'Percentile85', 'Stdev', 'Variance', 'Coeff of Variation']
    measure_names = ['v_foll [m/s]', 'a_foll [m/s2]', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway [s]', 'dX [m]']

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



    file.write('\n')


def initiate_trip_cf_enviro_stats_demo_behav_summary_file(file):
    file.write('Trip Car-Following & Enviro & Stats Summary File: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')

    file.write('Trip Information,')
    file.write(',,,,,,,,,,,,,,,,,,')
    file.write('Freeway,,Weather,,,,,,,Surface,,,,Visibility,,,Traffic,,,,,,Wiper Status,,,,,,,Statistics')
    file.write(',,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,')
    file.write('Driver Demographics and Behavior Traits')
    file.write('\n')

    file.write('Trip Number,Driver ID,Trip Length [min],Distance Traveled [km],')
    file.write('STAC Availability,')
    file.write('Time Bin,Day,Month,Year,')
    file.write('Start Time,Stop Time,Start Lat,Stop Lat,Start Long,Stop Long,')

    file.write('% Time in CF,Time in CF [min],No. CF Events,No. Targets Identified,No. Lead Targets Identified,')

    file.write('0-Not Freeway,1-Freeway,')
    file.write('1-Clear,2-Light Rain,3-Heavy Rain,4-Snowing,5-Fog,6-Sleeting,7-Mist/Light Rain,')
    file.write('1-Dry,2-Wet,3-Snowy,4-Icy,')
    file.write('1,2,3,')
    file.write('1-LOS A,2-LOS B,3-LOS C,4-LOS D,5-LOS E,6-LOS F,')
    file.write('Wiper: nan,Wiper: 0.0,Wiper: 1.0,Wiper: 2.0,Wiper: 3.0,Wiper: 254,Wiper: 255,')

    stats_operations_names = ['Mean', 'Max', 'Min', 'Median', 'Percentile85', 'Stdev', 'Variance', 'Coeff of Variation']
    measure_names = ['v_foll [m/s]', 'a_foll [m/s2]', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway [s]', 'dX [m]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    stats_operations_names = ['Min', 'Mean', 'Percentile15']
    measure_names = ['time to collision [s]']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    file.write('Driver ID,')
    file.write('Gender,Age Group,Ethnicity,Race,Education,Marital Status,Living Status,Work Status,')
    file.write('Household Population,Income,')
    file.write('Miles Driven Last Year,')

    file.write('Frequency of Tailgating,Frequency of Disregarding Speed Limit,Frequency of Aggressive Braking')

    file.write('\n')


def initiate_phase1_trip_enviro_stats_summary_file(file):
    """
    This summary file was generated for phase one data, but the data still need to be in the same
    format (columns the same as phase 2 data). This includes track 1 headway -- WHICH IS NOT ALWAYS THE
    LEAD VEHICLE HEADWAY!!!).
    :param file: save file
    :return: na
    """

    file.write('Trip Enviro & Stats Summary File: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))))
    file.write('\n')
    file.write('Trip Information,')
    file.write(',,,,,,,,,,,,')
    file.write('Freeway,,Weather,,,,,,,Surface,,,,Visibility,,,Traffic,,,,,,Wiper Status,,,,,,,Statistics')
    file.write('\n')

    file.write('Trip Number,Trip Length [min],Distance Traveled [km],')
    file.write('Time Bin,Day,Month,Year,')
    file.write('Start Time,Stop Time,Start Lat,Stop Lat,Start Long,Stop Long,')
    file.write('0-Not Freeway,1-Freeway,')
    file.write('1-Clear,2-Light Rain,3-Heavy Rain,4-Snowing,5-Fog,6-Sleeting,7-Mist/Light Rain,')
    file.write('1-Dry,2-Wet,3-Snowy,4-Icy,')
    file.write('1,2,3,')
    file.write('1-LOS A,2-LOS B,3-LOS C,4-LOS D,5-LOS E,6-LOS F,')
    file.write('Wiper: nan,Wiper: 0.0,Wiper: 1.0,Wiper: 2.0,Wiper: 3.0,Wiper: 254,Wiper: 255,')

    stats_operations_names = ['Mean', 'Max', 'Min', 'Median', 'Percentile85', 'Stdev', 'Variance', 'Coeff of Variation']
    measure_names = ['v_foll [m/s]', 'a_foll [m/s2]', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway [s]', 'dX [m]', 'time to collision [s]',
                        'track1_headway [s] NOT always lead vehicle']

    for i in range(len(stats_operations_names)):
        for j in range(len(measure_names)):
            if i == len(stats_operations_names)-1 and j == len(measure_names)-1:
                file.write(stats_operations_names[i] + ': ' + measure_names[j])
            else:
                file.write(stats_operations_names[i] + ': ' + measure_names[j] + ',')

    file.write('\n')



## NDS Summary File Append Functions ##
def append_to_ts_summary_file(point_collection,file,trip_no,observations=None):
    warnings.filterwarnings('ignore')  # Warnings regarding all "nan" values ignored
    split_point_collections = point_collection.segmentation(split_length=60)
    for i in range(len(split_point_collections)):
        file.write("{}_{},{},{},{},{},".format(trip_no, i + 1, trip_no, i + 1, 1,split_point_collections[i].dist_traveled()))

        time, day, month, year = split_point_collections[i].time_day_month_year()
        file.write("{},{},{},{},".format(time, day, month, year))

        start_time, stop_time = point_collection.start_stop_timestamp()
        start_lat, start_long = split_point_collections[i].start_lat_long()
        stop_lat, stop_long = split_point_collections[i].stop_lat_long()
        file.write("{},{},{},{},{},{},".format(start_time, stop_time, start_lat, stop_lat, start_long, stop_long))

        if observations != None:
            seg, length, time_start, time_stop, is_freeway, weather, surface, visibility, traffic = observations[i]
            file.write("{},{},{},{},{},".format(is_freeway, weather, surface, visibility, traffic))
        else:
            file.write("nan,nan,nan,nan,nan,")

        w1, w2, w3, w4, w5, w6, w7 = split_point_collections[i].percent_wiper_settings()
        file.write("{},{},{},{},{},{},{},".format(w1, w2, w3, w4, w5, w6, w7))

        measure_names = ['v_foll', 'a_foll', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway', 'dX']
        measure_values = []
        for j in range(len(measure_names)):
            temp = list()
            for k in range(split_point_collections[i].point_count()):
                temp.append(getattr(split_point_collections[i].list_of_data_points[k], measure_names[j]))
            measure_values.append(NewList(temp))
        del temp
        operations_names = ['mean','maximum','minimum','median','percentile','standard_deviation','variance',
                            'coeff_variation']
        for j in range(len(operations_names)):
            for k in range(len(measure_names)):
                temp = getattr(measure_values[k], operations_names[j])
                file.write("{},".format(temp()))
        del temp

        percent_cf = split_point_collections[i].percent_car_following()
        sec_consistent_cf = split_point_collections[i].time_continuous_car_following()
        no_lead_veh = len(split_point_collections[i].list_of_lead_targets())
        min_ttc = np.nanmin(split_point_collections[i].time_to_collision())
        file.write("{},{},{},{}".format(percent_cf,sec_consistent_cf,no_lead_veh,min_ttc))

        file.write("\n")


def append_to_cfs_summary_file(cf_collections,file,trip_no,stac_data_available,observations=None):

    if len(cf_collections) == 0:
        file.write("{}_0,{},0".format(trip_no,trip_no))
        file.write("\n")

    for i in range(len(cf_collections)):
        file.write("{}_{},{},{},{},".format(trip_no,i+1,trip_no,i+1,cf_collections[i].dist_traveled()))
        file.write("{},".format(cf_collections[i].time_elapsed()))

        file.write("{},".format(stac_data_available))

        time1, day, month, year = cf_collections[i].time_day_month_year()
        file.write("{},{},{},{},".format(time1, day, month, year))

        start_time, stop_time = cf_collections[i].start_stop_timestamp()
        start_lat, start_long = cf_collections[i].start_lat_long()
        stop_lat, stop_long = cf_collections[i].stop_lat_long()
        file.write("{},{},{},{},{},{},".format(start_time, stop_time, start_lat, stop_lat, start_long, stop_long))

        if observations != None:
            # Determine the starting and stopping one-minute segments
            start_segment = 0
            try:
                termination_condition = False
                while termination_condition is False:
                    if int(observations[start_segment][2]) < start_time and int(observations[start_segment][3]) > start_time:
                        termination_condition = True
                    else:
                        start_segment+=1

            except IndexError:
                start_segment = np.nan
            stop_segment = 0
            try:
                while int(observations[stop_segment][3]) < stop_time:
                    stop_segment+=1
            except IndexError:
                stop_segment = np.nan

            if start_segment == stop_segment:
                segment_list=[start_segment]
            elif np.isnan(start_segment) == True and np.isnan(stop_segment) == True:
                segment_list = []
                file.write("nan,nan,nan,nan,nan,")  # Bypass the writing to file below
            elif np.isnan(start_segment) == True:
                segment_list = [stop_segment]
            elif np.isnan(stop_segment) == True:
                segment_list = [start_segment]
            else:
                segment_list=range(start_segment,stop_segment+1)

            is_freeway = []
            weather = []
            surface = []
            visibility = []
            traffic = []
            for k in range(len(segment_list)):
                is_freeway_temp = int(observations[segment_list[k]][4])
                if is_freeway_temp not in is_freeway:
                    is_freeway.append(is_freeway_temp)
                del is_freeway_temp
                weather_temp = int(observations[segment_list[k]][5])
                if weather_temp not in weather:
                    weather.append(weather_temp)
                del weather_temp
                surface_temp = int(observations[segment_list[k]][6])
                if surface_temp not in surface:
                    surface.append(surface_temp)
                del surface_temp
                visibility_temp = int(observations[segment_list[k]][7])
                if visibility_temp not in visibility:
                    visibility.append(visibility_temp)
                del visibility_temp
                traffic_temp = int(observations[segment_list[k]][8])
                if traffic_temp not in traffic:
                    traffic.append(traffic_temp)
                del traffic_temp

            del segment_list

            reported_observations = [is_freeway,weather,surface,visibility,traffic]
            string_observations = []
            for k in range(len(reported_observations)):
                if len(reported_observations[k]) == 1:
                    string_observations.append("{}".format(reported_observations[k]))
                    string_observations[k] = string_observations[k].replace("[","")
                    string_observations[k] = string_observations[k].replace("]","")
                else:
                    string_observations.append("{}".format(reported_observations[k]))
                    string_observations[k] = string_observations[k].replace("[","")
                    string_observations[k] = string_observations[k].replace("]","")
                    string_observations[k] = string_observations[k].replace(",","_")

            del reported_observations

            for k in range(len(string_observations)):
                file.write("{},".format(string_observations[k]))
        else:
            file.write("nan,nan,nan,nan,nan,")

        w1, w2, w3, w4, w5, w6, w7 = cf_collections[i].percent_wiper_settings()
        file.write("{},{},{},{},{},{},{},".format(w1, w2, w3, w4, w5, w6, w7))

        measure_names = ['v_foll', 'a_foll', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway', 'dX', 'dV']
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
        file.write("{},{},{},{}".format(measure_values.minimum(), measure_values.mean(), measure_values.percentile(15),
                                        measure_values.median()))

        file.write("\n")


def append_to_var_exist_summary_file(point_collection,file,trip_no):

    file.write("{},".format(trip_no))
    variable_list = point_collection[0].__dict__.keys()
    variable_list.sort()
    variable_list.remove('data')
    variable_list.remove('index')
    variable_list.remove('stac_data_available')
    variable_list.remove('dV')
    counter = 0
    for variable in variable_list:
        percent = point_collection.percent_available(variable)
        if counter != len(variable_list) - 1:
            file.write("{},".format(percent))
        else:
            file.write("{}".format(percent))
        counter += 1
    file.write("\n")


def append_to_trip_cf_stats_summary_file(point_collection,cf_collections,driver_id,file,trip_no,stac_data_available):
    file.write('{},{},{},{},'.format(trip_no, driver_id, point_collection.time_elapsed() / 60,point_collection.dist_traveled()))

    file.write("{},".format(stac_data_available))

    time1, day, month, year = point_collection.time_day_month_year()
    file.write("{},{},{},{},".format(time1, day, month, year))

    start_time, stop_time = point_collection.start_stop_timestamp()
    start_lat, start_long = point_collection.start_lat_long()
    stop_lat, stop_long = point_collection.stop_lat_long()
    file.write("{},{},{},{},{},{},".format(start_time, stop_time, start_lat, stop_lat, start_long, stop_long))

    point_counter = 0
    for i in range(len(cf_collections)):
        point_counter+=cf_collections[i].point_count()

    file.write("{},{},{},{},{},{},".format(point_collection.percent_car_following(),point_collection.time_car_following(),
                                       len(cf_collections),point_counter/10./60.,len(point_collection.list_of_target_ids()),
                                       len(point_collection.list_of_lead_targets())))

    # Start using car-following collections - we are going to do the average of each of the operations/statistics for the cf events
    values_list = [[]for row in range(len(cf_collections))]
    counter = 0
    for cf_collection in cf_collections:
        operations_names = ['mean', 'maximum', 'minimum', 'median', 'standard_deviation', 'variance', 'percentile']
        measure_names = ['v_foll', 'a_foll', 'headway', 'dX', 'dV', 'v_lead','a_lead','time_to_collision']

        measure_values = []
        for j in range(len(measure_names)):
            temp = list()
            for k in range(cf_collection.point_count()):
                temp.append(getattr(cf_collection.list_of_data_points[k], measure_names[j]))
            measure_values.append(NewList(temp))
        del temp

        for j in range(len(operations_names)):
            for k in range(len(measure_names)):
                temp = getattr(measure_values[k], operations_names[j])
                values_list[counter].append(temp())
        counter+=1

    # Transpose Values List
    array = np.array(values_list)
    array = array.transpose()

    # Take the average of each column...
    for i in range(len(array)):
        if i == len(array)-1:
            file.write("{}".format(np.nanmean(array[i])))
        else:
            file.write("{},".format(np.nanmean(array[i])))

    file.write("\n")


def append_to_trip_enviro_stats_summary_file(point_collection,file,trip_no,observations=None):
    file.write('{},{},{},'.format(trip_no, point_collection.time_elapsed() / 60, point_collection.dist_traveled()))

    time1, day, month, year = point_collection.time_day_month_year()
    file.write("{},{},{},{},".format(time1, day, month, year))

    start_time, stop_time = point_collection.start_stop_timestamp()
    start_lat,start_long = point_collection.start_lat_long()
    stop_lat,stop_long = point_collection.stop_lat_long()
    file.write("{},{},{},{},{},{},".format(start_time,stop_time,start_lat,stop_lat,start_long,stop_long))

    if observations is not None:
        # Is Freeway: 0,1
        index = 4
        options = [0,1]
        count = [0,0]
        percent = [0,0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i]/float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Weather Condition: 1,2,3,4,5,6,7
        index = 5
        options = [1,2,3,4,5,6,7]
        count = [0,0,0,0,0,0,0]
        percent = [0,0,0,0,0,0,0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i]/float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Surface Condition: 1,2,3,4
        index = 6
        options = [1,2,3,4]
        count = [0,0,0,0]
        percent = [0,0,0,0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i]/float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Visibility: 1,2,3
        index = 7
        options = [1,2,3]
        count = [0,0,0]
        percent = [0,0,0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i]/float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Traffic Condition: 1,2,3,4,5,6
        index = 8
        options = [1,2,3,4,5,6]
        count = [0,0,0,0,0,0]
        percent = [0,0,0,0,0,0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i]/float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

    else:
        file.write('nan,nan,')
        file.write('nan,nan,nan,nan,nan,nan,nan,')
        file.write('nan,nan,nan,nan,')
        file.write('nan,nan,nan,')
        file.write('nan,nan,nan,nan,nan,nan,')

    # Wipers
    percent = point_collection.percent_wiper_settings()
    for i in range(len(percent)):
        file.write('{},'.format(percent[i]))

    # Statistics
    measure_names = ['v_foll', 'a_foll', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway', 'dX', 'vtti_accel_y',
                         'vtti_light_level', 'vtti_head_confidence',
                         'vtti_head_position_x','vtti_head_position_x_baseline',
                         'vtti_head_position_y', 'vtti_head_position_y_baseline',
                         'vtti_head_position_z', 'vtti_head_position_z_baseline',
                         'vtti_head_rotation_x', 'vtti_head_rotation_x_baseline',
                         'vtti_head_rotation_y', 'vtti_head_rotation_y_baseline',
                         'vtti_head_rotation_z', 'vtti_head_rotation_z_baseline']
    measure_values = []
    for j in range(len(measure_names)):
        temp = list()
        for k in range(point_collection.point_count()):
            temp.append(getattr(point_collection.list_of_data_points[k], measure_names[j]))
        measure_values.append(NewList(temp))
    del temp
    operations_names = ['mean', 'maximum', 'minimum', 'median', 'percentile', 'standard_deviation','variance','coeff_variation']
    for j in range(len(operations_names)):
        for k in range(len(measure_names)):
            temp = getattr(measure_values[k], operations_names[j])
            file.write("{},".format(temp()))

    # Time to Collision
    measure_values = NewList(point_collection.time_to_collision())
    file.write("{},{},{}".format(measure_values.minimum(),measure_values.mean(),measure_values.percentile(15)))

    file.write("\n")


def append_to_trip_cf_enviro_stats_summary_file(point_collection,cf_collections,file,trip_no,
                                                stac_data_available,observations=None):
    file.write('{},{},{},'.format(trip_no, point_collection.time_elapsed() / 60,point_collection.dist_traveled()))

    file.write("{},".format(stac_data_available))

    time1, day, month, year = point_collection.time_day_month_year()
    file.write("{},{},{},{},".format(time1, day, month, year))

    start_time, stop_time = point_collection.start_stop_timestamp()
    start_lat, start_long = point_collection.start_lat_long()
    stop_lat, stop_long = point_collection.stop_lat_long()
    file.write("{},{},{},{},{},{},".format(start_time, stop_time, start_lat, stop_lat, start_long, stop_long))

    file.write("{},{},{},{},{},".format(point_collection.percent_car_following(),point_collection.time_car_following(),
                                       len(cf_collections),len(point_collection.list_of_target_ids()),
                                       len(point_collection.list_of_lead_targets())))

    if observations is not None:
        # Is Freeway: 0,1
        index = 4
        options = [0, 1]
        count = [0, 0]
        percent = [0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Weather Condition: 1,2,3,4,5,6,7
        index = 5
        options = [1, 2, 3, 4, 5, 6, 7]
        count = [0, 0, 0, 0, 0, 0, 0]
        percent = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Surface Condition: 1,2,3,4
        index = 6
        options = [1, 2, 3, 4]
        count = [0, 0, 0, 0]
        percent = [0, 0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Visibility: 1,2,3
        index = 7
        options = [1, 2, 3]
        count = [0, 0, 0]
        percent = [0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Traffic Condition: 1,2,3,4,5,6
        index = 8
        options = [1, 2, 3, 4, 5, 6]
        count = [0, 0, 0, 0, 0, 0]
        percent = [0, 0, 0, 0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

    else:
        file.write('nan,nan,')
        file.write('nan,nan,nan,nan,nan,nan,nan,')
        file.write('nan,nan,nan,nan,')
        file.write('nan,nan,nan,')
        file.write('nan,nan,nan,nan,nan,nan,')

    # Wipers
    percent = point_collection.percent_wiper_settings()
    for i in range(len(percent)):
        file.write('{},'.format(percent[i]))

    # Statistics
    measure_names = ['v_foll', 'a_foll', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway', 'dX']
    measure_values = []
    for j in range(len(measure_names)):
        temp = list()
        for k in range(point_collection.point_count()):
            temp.append(getattr(point_collection.list_of_data_points[k], measure_names[j]))
        measure_values.append(NewList(temp))
    del temp
    operations_names = ['mean', 'maximum', 'minimum', 'median', 'percentile', 'standard_deviation', 'variance',
                        'coeff_variation']
    for j in range(len(operations_names)):
        for k in range(len(measure_names)):
            temp = getattr(measure_values[k], operations_names[j])
            file.write("{},".format(temp()))

    # Time to Collision
    measure_values = NewList(point_collection.time_to_collision())
    file.write("{},{},{}".format(measure_values.minimum(), measure_values.mean(), measure_values.percentile(15)))

    file.write("\n")


def append_to_trip_cf_enviro_stats_demo_behav_summary_file(point_collection,cf_collections,file,trip_no,driver_id,stac_data_available,
                                observations=None,demographics_data=None,behavior_data=None):
    file.write('{},{},{},{},'.format(trip_no, driver_id, point_collection.time_elapsed() / 60,point_collection.dist_traveled()))

    file.write("{},".format(stac_data_available))

    time1, day, month, year = point_collection.time_day_month_year()
    file.write("{},{},{},{},".format(time1, day, month, year))

    start_time, stop_time = point_collection.start_stop_timestamp()
    start_lat, start_long = point_collection.start_lat_long()
    stop_lat, stop_long = point_collection.stop_lat_long()
    file.write("{},{},{},{},{},{},".format(start_time, stop_time, start_lat, stop_lat, start_long, stop_long))

    file.write("{},{},{},{},{},".format(point_collection.percent_car_following(),point_collection.time_car_following(),
                                       len(cf_collections),len(point_collection.list_of_target_ids()),
                                       len(point_collection.list_of_lead_targets())))

    if observations is not None:
        # Is Freeway: 0,1
        index = 4
        options = [0, 1]
        count = [0, 0]
        percent = [0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Weather Condition: 1,2,3,4,5,6,7
        index = 5
        options = [1, 2, 3, 4, 5, 6, 7]
        count = [0, 0, 0, 0, 0, 0, 0]
        percent = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Surface Condition: 1,2,3,4
        index = 6
        options = [1, 2, 3, 4]
        count = [0, 0, 0, 0]
        percent = [0, 0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Visibility: 1,2,3
        index = 7
        options = [1, 2, 3]
        count = [0, 0, 0]
        percent = [0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Traffic Condition: 1,2,3,4,5,6
        index = 8
        options = [1, 2, 3, 4, 5, 6]
        count = [0, 0, 0, 0, 0, 0]
        percent = [0, 0, 0, 0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

    else:
        file.write('nan,nan,')
        file.write('nan,nan,nan,nan,nan,nan,nan,')
        file.write('nan,nan,nan,nan,')
        file.write('nan,nan,nan,')
        file.write('nan,nan,nan,nan,nan,nan,')

    # Wipers
    percent = point_collection.percent_wiper_settings()
    for i in range(len(percent)):
        file.write('{},'.format(percent[i]))

    # Statistics
    measure_names = ['v_foll', 'a_foll', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                         'vtti_wiper', 'headway', 'dX']
    measure_values = []
    for j in range(len(measure_names)):
        temp = list()
        for k in range(point_collection.point_count()):
            temp.append(getattr(point_collection.list_of_data_points[k], measure_names[j]))
        measure_values.append(NewList(temp))
    del temp
    operations_names = ['mean', 'maximum', 'minimum', 'median', 'percentile', 'standard_deviation', 'variance',
                        'coeff_variation']
    for j in range(len(operations_names)):
        for k in range(len(measure_names)):
            temp = getattr(measure_values[k], operations_names[j])
            file.write("{},".format(temp()))

    # Time to Collision
    measure_values = NewList(point_collection.time_to_collision())
    file.write("{},{},{},".format(measure_values.minimum(), measure_values.mean(), measure_values.percentile(15)))

    # Driver Demographics
    if demographics_data == None:
        for j in range(12):
            file.write('{},'.format(np.nan))
    else:
        # file.write('Gender,Age Group,Ethnicity,Race,Education,Marital Status,Living Status,Work Status,')
        file.write('{},'.format(demographics_data[0]))
        file.write('{},{},{},'.format(demographics_data[1],demographics_data[2],demographics_data[3]))
        file.write('{},{},{},'.format(demographics_data[4],demographics_data[6],demographics_data[7]))
        file.write('{},{},'.format(demographics_data[8],demographics_data[10]))
        # file.write('Income,Household Population,')
        file.write('{},{},'.format(demographics_data[11],demographics_data[12]))
        # file.write('Miles Driven Last Year,')
        file.write('{},'.format(demographics_data[44]))

    # Driver Behavior
    if behavior_data == None:
        file.write("{},{},{}".format(np.nan,np.nan,np.nan))
    else:
        # file.write('Frequency of Tailgating,Frequency of Disregarding Speed Limit,Frequency of Aggressive Braking')
        file.write('{},{},{}'.format(behavior_data[3],behavior_data[12],behavior_data[24]))

    file.write("\n")


def append_to_phase1_trip_enviro_stats_summary_file(point_collection,file,trip_no,observations=None):
    file.write('{},{},{},'.format(trip_no, point_collection.time_elapsed() / 60, point_collection.dist_traveled()))

    time1, day, month, year = point_collection.time_day_month_year()
    file.write("{},{},{},{},".format(time1, day, month, year))

    start_time, stop_time = point_collection.start_stop_timestamp()
    start_lat, start_long = point_collection.start_lat_long()
    stop_lat, stop_long = point_collection.stop_lat_long()
    file.write("{},{},{},{},{},{},".format(start_time, stop_time, start_lat, stop_lat, start_long, stop_long))

    if observations is not None:
        # Is Freeway: 0,1
        index = 4
        options = [0, 1]
        count = [0, 0]
        percent = [0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Weather Condition: 1,2,3,4,5,6,7
        index = 5
        options = [1, 2, 3, 4, 5, 6, 7]
        count = [0, 0, 0, 0, 0, 0, 0]
        percent = [0, 0, 0, 0, 0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Surface Condition: 1,2,3,4
        index = 6
        options = [1, 2, 3, 4]
        count = [0, 0, 0, 0]
        percent = [0, 0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Visibility: 1,2,3
        index = 7
        options = [1, 2, 3]
        count = [0, 0, 0]
        percent = [0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

        # Traffic Condition: 1,2,3,4,5,6
        index = 8
        options = [1, 2, 3, 4, 5, 6]
        count = [0, 0, 0, 0, 0, 0]
        percent = [0, 0, 0, 0, 0, 0]
        for i in range(len(observations)):
            for j in range(len(options)):
                if int(observations[i][index]) == options[j]:
                    count[j] += 1
        for i in range(len(count)):
            percent[i] = count[i] / float(len(observations))
        for i in range(len(percent)):
            file.write('{},'.format(percent[i]))

    else:
        file.write('nan,nan,')
        file.write('nan,nan,nan,nan,nan,nan,nan,')
        file.write('nan,nan,nan,nan,')
        file.write('nan,nan,nan,')
        file.write('nan,nan,nan,nan,nan,nan,')

    # Wipers
    percent = point_collection.percent_wiper_settings()
    for i in range(len(percent)):
        file.write('{},'.format(percent[i]))

    # Statistics
    measure_names = ['v_foll', 'a_foll', 'vtti_gyro_z', 'vtti_lane_distance_off_center',
                                     'vtti_wiper', 'headway', 'dX', 'time_to_collision','track1_headway']
    measure_values = []
    for j in range(len(measure_names)):
        temp = list()
        for k in range(point_collection.point_count()):
            temp.append(getattr(point_collection.list_of_data_points[k], measure_names[j]))
        measure_values.append(NewList(temp))
    del temp
    operations_names = ['mean', 'maximum', 'minimum', 'median', 'percentile', 'standard_deviation', 'variance',
                        'coeff_variation']
    for j in range(len(operations_names)):
        for k in range(len(measure_names)):
            temp = getattr(measure_values[k], operations_names[j])
            if j == len(operations_names)-1 and k == len(measure_names)-1:
                file.write("{}".format(temp()))
            else:
                file.write("{},".format(temp()))

    file.write("\n")
