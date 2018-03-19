from __future__ import division
import csv
import os
import numpy as np
from classes import WyNdsDataPoint
from classes import WyNdsReducedDataPoint
from classes import WzDataPoint
from classes import WyNdsPointCollection
from classes import WyNdsReducedPointCollection
from classes import WzPointCollection


"""
/*******************************************************************
Basic functions used to process and analyze CF Data, in 
conjunction with classes.py

Author: Britton Hammit
E-mail: bhammit1@gmail.com
Date: 10-04-2017
********************************************************************/
"""

### NDS Functions ###
def import_trip_ids(path,file_name,log_file=None):
    trip_num = list()
    file = open(os.path.join(path, file_name), 'r')
    for line in file:
        trip_num.append(line.strip().split(','))
    file.close()
    print "Number of Trips for Analysis: {}".format(len(trip_num))
    if log_file is not None:
        log_file.write("Number of Trips for Analysis: {}".format(len(trip_num)))
        log_file.write("\n")
        log_file.write("\n")
    return trip_num


def check_files_exist(trip_numbers,open_nds_data_path,nds_file_name_start):
    # Check validity of each trip number before starting analysis
    file_error_count = 0
    for i in range(len(trip_numbers)):
        try:
            file = open(os.path.join(open_nds_data_path, '{}_{}.csv'.format(nds_file_name_start,trip_numbers[i][0])), 'r')
            """
            try:
                temp = np.genfromtxt(file, delimiter=',', skip_header=1, missing_values=np.nan)
            except ValueError:
                print "ERROR VALUE ERROR - INCORRECT NUMBER OF COLUMNS: {}_{}".format(nds_file_name_start,trip_numbers[i][0])
                file_error_count += 1
            del temp
            """
            file.close()
        except IOError:
            print "ERROR FILE DOES NOT EXIST: {}_{}".format(nds_file_name_start,trip_numbers[i][0])
            file_error_count += 1
    if file_error_count > 0:  # causes the program to come to a stop IF one trip number is invalid
        raise SystemError  # Stop running script if error detected


def initialize_save_path(save_path):
    # If save_path doesn't already exist, it will be created
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def import_wy_nds(nds_file_name,path,stac=False):
    # Import NDS Data as DATA POINTS
    nds_file = open(os.path.join(path,nds_file_name), 'r')  # Open NDSfile
    nds_data = np.genfromtxt(nds_file, delimiter=',',skip_header=1,missing_values=np.nan)
    # Import data such that blank entries are given 'nan' values
    nds_file.close()

    data_points = list()
    for i in range(len(nds_data)):
        data_points.append(WyNdsDataPoint(nds_data[i]))
    del nds_data
    if stac is True:
        return data_points
    else:
        return WyNdsPointCollection(data_points)


def import_nds_stac(stac_file_name, path):
    stac_file = open(os.path.join(path,stac_file_name), 'r')  # Open NDSfile
    # Import data such that blank entries are given 'nan' values
    stac_data = np.genfromtxt(stac_file, delimiter=',',skip_header=1,missing_values=np.nan)
    if len(stac_data) == 0:
        print "      "+"*STAC File Empty"
        raise IOError

    stac_file.close()
    return stac_data


def import_wy_nds_stac(nds_file_name,nds_path,stac_file_name,stac_path):

    def time_interpollation(time,value,nds_time):
        new_value = [np.nan for i in range(len(nds_time))]

        j = 0
        for i in range(len(nds_time)):
            while time[j]<nds_time[i] and j<len(time)-1:
                if time[j]<nds_time[i] and time[j+1]>nds_time[i]:
                    new_value[i] = (value[j+1]-value[j])/(time[j+1]-time[j])*(nds_time[i]-time[j])+value[j]
                j+=1

        # Get rid of non values by averaging existing values
        for i in range(len(new_value)-2):
            if np.isfinite(new_value[i+1]) == False:
                new_value[i+1] = np.nanmean([new_value[i],new_value[i+2]])
        return new_value

    data_points = import_wy_nds(nds_file_name,nds_path,stac=True)  # Point Collection!
    stac_data = import_nds_stac(stac_file_name, stac_path)

    new_time_nds = list()
    for i in range(len(data_points)):
        new_time_nds.append(data_points[i].vtti_time_stamp)

    time_stac = list()
    for i in range(len(stac_data)):
        time_stac.append(stac_data[i][0])

    value_stac_dV = list()
    for i in range(len(stac_data)):
        value_stac_dV.append(stac_data[i][6])

    # Separate stac based on track
    track_list = [1,2,3,4,5,6,7,8]
    track_value_list_stac = [list() for i in range(len(track_list))]
    track_time_list_stac = [list() for i in range(len(track_list))]

    for i in range(len(stac_data)):
        for j in range(len(track_list)):
            if track_list[j]==stac_data[i][1]:
                track_value_list_stac [j].append(stac_data[i][10])
                track_time_list_stac[j].append(stac_data[i][0])

    track_value_new_list = [list() for i in range(len(track_list))]
    for i in range(len(track_list)):
        if len(track_time_list_stac[i]) != 0:  # Accounting for tracks with no lead vehicles
            track_value_new_list[i] = time_interpollation(track_time_list_stac[i],track_value_list_stac[i],new_time_nds)

    for i in range(len(track_value_new_list)):
        if len(track_value_new_list[i]) != 0:  # Accounting for tracks with no lead vehicles
            if i+1 == 1:
                for j in range(len(data_points)):
                    data_points[j].track1_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 2:
                for j in range(len(data_points)):
                    data_points[j].track2_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 3:
                for j in range(len(data_points)):
                    data_points[j].track3_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 4:
                for j in range(len(data_points)):
                    data_points[j].track4_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 5:
                for j in range(len(data_points)):
                    data_points[j].track5_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 6:
                for j in range(len(data_points)):
                    data_points[j].track6_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 7:
                for j in range(len(data_points)):
                    data_points[j].track7_x_vel_processed = track_value_new_list[i][j]
            elif i+1 == 8:
                for j in range(len(data_points)):
                    data_points[j].track8_x_vel_processed = track_value_new_list[i][j]

    for i in range(len(data_points)):
        # Change variable indicating that STAC data is available and should be used.
        data_points[i].stac_data_available = True
        data_points[i].reset_super_attributes()

    return WyNdsPointCollection(data_points)


def import_wy_nds_reduced(file_name,path):
    """
    Import the reduced data file generated for a CF Event or Timeseries Segment.
    :param file_name:
    :param path:
    :return:
    """
    reduced_file = open(os.path.join(path,file_name),'r')
    reduced_file.next()
    data_points = list()
    for line in reduced_file:
        data_points.append(WyNdsReducedDataPoint(line.strip().split(',')))
    reduced_file.close()
    return WyNdsReducedPointCollection(data_points)


def import_obs_file(filename,path):
    file = open(os.path.join(path,filename),'r')
    temp = list()
    for i in range(19):
        file.next()  # Skip Headers
    for line in file:
        try:
            if type(int(float(line[0]))) == int:
                temp.append(line.strip().split(','))
        except ValueError:
            continue
    file.close()
    return temp


def generate_cf_collections_nds(nds_file_name_start, trip_no, nds_path, stac_path, min_cf_time, max_cf_dist, min_speed):
    """
    Function to generate car-following events/collections from NDS data
    # NOTE Moving average filter always running with resolution = 14!
    :param nds_file_name_start: file name style: Event_ID or Trip_ID
    :param trip_no:
    :param nds_path:
    :param stac_path:
    :param min_cf_time:
    :param max_cf_dist:
    :param min_speed:
    :return: list of CF Collections (PointCollection - Class) AND stac_data_available indicator variable
    """
    NDSfile = '{}_{}.csv'.format(nds_file_name_start, trip_no)
    STACfile = '{}_stac_{}.csv'.format(nds_file_name_start, trip_no)

    try:
        point_collection = import_wy_nds_stac(nds_file_name=NDSfile,
                                              nds_path=nds_path, stac_file_name=STACfile,
                                              stac_path=stac_path)
        stac_data_available = True
    except IOError:
        point_collection = import_wy_nds(nds_file_name=NDSfile,path=nds_path)
        stac_data_available = False
        print "    " + "*STAC Radar Data Not Available"
    cf_collections = point_collection.car_following_event_extraction(min_cf_time=min_cf_time,
                                                                     max_cf_dist=max_cf_dist,
                                                                     min_speed=min_speed)

    for collection in cf_collections:
        collection.moving_average(resolution=14)

    return point_collection,cf_collections,stac_data_available


def get_driver_id(data_dictionary_file_name,path,trip_no):
    data_dictionary_file = open(os.path.join(path,data_dictionary_file_name),'r')
    data_dictionary_file.next()
    data_dictionary_list = list()
    for line in data_dictionary_file:
        data_dictionary_list.append(line.strip().split(","))
    data_dictionary_file.close()
    del line

    # Column Indicies for Data Dictionary File
    i_driver_id = 0
    i_adverse_event = 1
    i_clear_1_event = 5
    i_clear_2_event = 9
    col_indicies = [i_adverse_event,i_clear_1_event,i_clear_2_event]

    driver_id = np.nan
    for i in range(len(data_dictionary_list)):
        for j in col_indicies:
            if data_dictionary_list[i][j] == trip_no:
                driver_id = int(data_dictionary_list[i][i_driver_id])

    return driver_id


def get_behavior_survey_data(file_name,path,driver_id):
    file = open(os.path.join(path, file_name), 'r')
    file.next()
    behav_data_list = list()
    reader = csv.reader(file, skipinitialspace=True)
    for line in reader:
        behav_data_list.append(line)

    # Column Indicies for Behavior Survey
    i_driver_id = 0

    behav_data = None
    for i in range(len(behav_data_list)):
        if int(behav_data_list[i][i_driver_id]) == driver_id:
            behav_data = behav_data_list[i]

    for i in range(len(behav_data)):
        behav_data[i] = behav_data[i].replace(",",";")

    return behav_data


def get_demographics_survey_data(file_name,path,driver_id):
    file = open(os.path.join(path,file_name),'r')
    file.next()
    demo_data_list = list()
    reader = csv.reader(file, skipinitialspace=True)
    for line in reader:
        demo_data_list.append(line)

    # Column Indicies for Demographics Survey
    i_driver_id = 0

    demo_data = None
    for i in range(len(demo_data_list)):
        if int(demo_data_list[i][i_driver_id]) == driver_id:
            demo_data = demo_data_list[i]
    for i in range(len(demo_data)):
        demo_data[i] = demo_data[i].replace(",",";")

    return demo_data


### FHWA WZ Functions ###
def import_fhwa_wz(file_name,path):
    """
    Imports FHWA Work Zone data as WZ Data Points and into WZ Point Collections
    :param file_name:
    :param path:
    :return: PointCollection containing data from input file
    """

    file = open(os.path.join(path,file_name), 'r')  # Open Vehicle Data file
    for i in range(14):  # Number of header lines
        file.next()  # Header Line
    data = []
    for line in file:
        data.append(line.strip().split(','))
    file.close()

    # Convert data from Float to Integer & Create Data Points
    list_of_data_points = list()
    for i in range(len(data)):
        for j in range(len(data[i])):
            try:
                data[i][j] = float(data[i][j])
            except ValueError:
                data[i][j] = np.nan
        list_of_data_points.append(WzDataPoint(data=data[i]))

    point_collection = WzPointCollection(list_of_data_points=list_of_data_points)

    return point_collection


### Basic Calibration Functions ###
def calc_RMSE(pred_list, act_list):
    """
    Calculate the root mean square error between a list of predicted points and actual points.
    :param pred_list: List of predicted points
    :param act_list: List of actual points
    :return: RMSE: Float of the RMSE describing the two input data lists
    """

    diff_square = list()
    nan_count = 0
    for i in xrange(len(pred_list)):
        temp_diff_square = (pred_list[i] - act_list[i])**2
        if np.isnan(temp_diff_square) == True:
            nan_count += 1
        diff_square.append(temp_diff_square)
        del temp_diff_square
    sum_diff_square = np.nansum(diff_square)
    RMSE = np.sqrt(sum_diff_square/(len(pred_list)-nan_count))

    del diff_square,nan_count,sum_diff_square

    return RMSE


def calc_r(x,y):
    """
    Correlation Coefficient from Cuiffo Report
    :param x: list of values
    :param y: list of values
    :return: Correlation Coefficient
    """
    std_x = np.nanstd(x)
    std_y = np.nanstd(y)
    mean_x = np.nanmean(x)
    mean_y = np.nanmean(y)

    sum_list = list()
    for i in range(len(x)):
        temp = (x[i]-mean_x)*(y[i]-mean_y)/(std_x*std_y)
        sum_list.append(temp)

    return np.nansum(sum_list)/(len(x)-1)


def plot_cf_collections(cf_collections, trip_no, trip_save_path):
    """
    Plots car-following collections
    :param cf_collections:
    :param trip_no:
    :param trip_save_path:
    :return:
    """
    counter = 0
    for collection in cf_collections:
        label = "{}_{}".format(trip_no, counter)
        plot_fig = collection.cf_summary_plot(trip_no=label)
        plot_fig.savefig(os.path.join(trip_save_path, '{}_{}_cf_plot'.format(trip_no, counter)))
        counter += 1
    return None










