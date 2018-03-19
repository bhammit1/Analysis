import os
import numpy as np
from tools.functions_basic import import_trip_ids
from tools.functions_basic import check_files_exist
from tools.functions_basic import initialize_save_path
from tools.functions_basic import import_wy_nds
from tools.functions_basic import import_wy_nds_stac
from tools.functions_basic import import_obs_file
from tools.functions_basic import get_driver_id
from tools.functions_basic import get_behavior_survey_data
from tools.functions_basic import get_demographics_survey_data
from tools.functions_wyNdsTool import initiate_ts_summary_file
from tools.functions_wyNdsTool import initiate_cfs_summary_file
from tools.functions_wyNdsTool import initiate_var_exist_summary_file
from tools.functions_wyNdsTool import initiate_trip_cf_stats_summary_file
from tools.functions_wyNdsTool import initiate_trip_enviro_stats_summary_file
from tools.functions_wyNdsTool import initiate_trip_cf_enviro_stats_summary_file
from tools.functions_wyNdsTool import initiate_trip_cf_enviro_stats_demo_behav_summary_file
from tools.functions_wyNdsTool import initiate_phase1_trip_enviro_stats_summary_file
from tools.functions_wyNdsTool import append_to_ts_summary_file
from tools.functions_wyNdsTool import append_to_cfs_summary_file
from tools.functions_wyNdsTool import append_to_var_exist_summary_file
from tools.functions_wyNdsTool import append_to_trip_cf_stats_summary_file
from tools.functions_wyNdsTool import append_to_trip_enviro_stats_summary_file
from tools.functions_wyNdsTool import append_to_trip_cf_enviro_stats_summary_file
from tools.functions_wyNdsTool import append_to_trip_cf_enviro_stats_demo_behav_summary_file
from tools.functions_wyNdsTool import append_to_phase1_trip_enviro_stats_summary_file

"""
/*******************************************************************
Wyoming NDS Data Analysis Tool Version 2 -- Main File

Author: Britton Hammit
E-mail: bhammit1@gmail.com
Last Update: 03-16-2018
-----------------------------------------------------------------------

Necessary Script Files:
-----------------------------------------------------------------------
1. wy_nds_tool_v2-4.py
2. //tools
   - functions_basic.py
   - functions_wyNdsTool.py

Functionalities:
-----------------------------------------------------------------------
Single Trip Functions

General:
1. Save reduced output file for full trip (.csv)

Time Segmentation (TS):
2. TS: Save split time series files (.csv)
3. TS: Create manual video observation template (.csv)
4. TS: Generate descriptive statistics (.csv)
5. TS: Generate time series plots (.png)

Car-Following Segmentation (CFS):
6. CFS: Save time series files (.csv)
7. 	CFS: Generate individual car-following plots (.png)
8.	CFS: Generate individual summary car-following plot (.png)
9.	CFS: Generate time series plots (.png)

Multi-Trip Functions

Trip Summary Files:
10.	Variable existence summary file (.csv)
11.	Enviro & Statistics summary file (.csv)
12.	Car-Following summary file (.csv)
13.	Enviro & Statistics & Car-Following summary file (.csv)
14.	Enviro & Statistics & Car-Following & Demographics & Behavior summary file (.csv)

Time Segmentation (TS) Summary Files:
15.	TS: Summary file listing each one-minute segment for all trips (.csv)

Car-Following Segmentation (CFS) Summary Files:
16.	CFS: Summary file listing each car-following event for all trips (.csv)

********************************************************************/
"""

"""
Global Variables - CHANGE THESE NAMES/ LOCATIONS AS NECESSARY
"""
##### Path Variables where input files and NDS data #####
# Global Inputs:
open_global_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\01-Global_Input_Files'
# NDS .csv Data Files:
open_nds_data_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 a.Nds_Stac_Files'
open_stac_data_path = open_nds_data_path
# Result Location:
save_path_start = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.03.16 WyNdsTool Update'
# If Completed Observations are not saved in trip folder
completed_observations_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\02-Manual_Video_Observations'
# Must use the "r" before path because of the number "00-" in the path
survey_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\02-NDS_Dictionaries'

##### Input Files: Trip IDs used for processing - saved in "Global Inputs" file #####
trip_no_for_processing = 'fhwa_ALL_trip_ids_lasthalf.csv'
nds_file_name_start = 'Event_ID'  # Change between Event_ID and File_ID as needed depending on the NDS Time series input files
behavior_survey_file_name = 'DrivingBehaviorQuestionnaire_UW.csv'
demographics_survey_file_name = 'DriverDemographicQuestionnaire_UW.csv'
data_dictionary_file_name = 'RainClear_Events_UW_A.csv'

##### Car-Following Settings #####
min_cf_time = 20  # s
max_cf_dist = 60  # m
min_speed = 1  # m/s

##### Moving Average Filter Settings for Radar Data #####
moving_average_filter = True

##### Summary File Name Start (Default - Date) #####
date = '2018-03-16'

"""
Methods to Run
"""
##### Single Trip Functions: #####
# Saves new time series file for trip with relevant parameters (including additional cf parameters)
save_reduced_output_file_for_full_trip = False

##### Time Segmentation (TS)#####
# Saves individual time series files for each one-minute segment
TS_save_split_time_series_files = False
# Saves manual video observation file for the trip - to summarize each one-minute segment
TS_create_manual_video_observation_template = False
# Saves summary time series statistics for each one-minute segment
TS_generate_descriptive_statistics = False
# Saves summary time series plots for each one-minute segment
TS_generate_time_series_plots = False

##### Car-Following Segmentation (CFS)#####
# Saves reduced time series files for each car-following event
CFS_save_time_series_file = False
# Saves car-following plots for each car-following event meeting the criteria
CFS_generate_individual_cf_plots = False
# Saves single summary car-following plot for each car-following event meeting the criteria
CFS_generate_cf_plot = False
# Saves summary time series plots for each car-following event
CFS_generate_time_series_plots = False

##### Multi-Trip Functions: #####
# Summary file describing variable availability for all trips
generate_variable_exist_summary_file = False

# Summary file describing the manual environmental observations and time series statistics for all trips
# 3/16/2018 Added variables: accel_y, light_level, head_confidence, head_position
generate_trip_enviro_stats_summary_file = True

# Summary file describing car-following events stats for all trips (mean of the values for each CFE)
generate_trip_cf_stats_summary_file = False

# Summary file describing car-following events, manual environmental observations, and time series stats for all trips
generate_trip_cf_enviro_stats_summary_file = False

# Summary file describing car-following events, manual environmental observations, time series stats, driver
# demographics, and select behavioral tendencies from self-reported survey for all trips
generate_trip_cf_enviro_stats_demo_behav_summary_file = False

# Summary file describing each one-minute segment for all trips
generate_TS_summary_file = False

# Summary file describing each car-following event for all trips
generate_CFS_summary_file = False

# PHASE 1 summary file - trip_enviro_stats_summary_file ++ track1_headway - NOT always lead vehicle!
generate_phase1_trip_enviro_stats_summary_file = False

##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
############################################ No inputs past this point ###########################################
##################################################################################################################
##################################################################################################################
##################################################################################################################
##################################################################################################################

"""
Initiations
"""
# File Names
var_exist_summary_file_name = '{}_var_exist_summary.csv'.format(date)
trip_enviro_stats_summary_file_name = '{}_trip_enviro_stats_summary.csv'.format(date)
trip_cf_stats_summary_file_name = '{}_trip_cf_summary.csv'.format(date)
trip_cf_enviro_stats_summary_file_name = '{}_trip_cf_enviro_stats_summary.csv'.format(date)
trip_cf_enviro_stats_demo_behav_summary_file_name = '{}_trip_cf_enviro_stats_demo_behav_summary.csv'.format(date)
TS_summary_file_name = '{}_TS_summary.csv'.format(date)
CFS_summary_file_name = '{}_CFS_summary.csv'.format(date)
phase1_trip_enviro_stats_summary_file_name = '{}_ph1_trip_enviro_stats_summary.csv'.format(date)

# Read in trip numbers from file
trip_num = import_trip_ids(open_global_input_path,trip_no_for_processing)

# Check validity of each trip number before starting analysis
check_files_exist(trip_num,open_nds_data_path,nds_file_name_start)

# Initialize multi-trip summary files
if generate_variable_exist_summary_file is True:
    var_exist_summary_file = open(os.path.join(save_path_start, var_exist_summary_file_name), 'a')
    NDSfile = '{}_{}.csv'.format(nds_file_name_start,trip_num[0][0])
    point_collection = import_wy_nds(nds_file_name=NDSfile, path=open_nds_data_path)
    initiate_var_exist_summary_file(file=var_exist_summary_file, point_collection=point_collection)
    del NDSfile
    del point_collection

if generate_trip_enviro_stats_summary_file is True:
    trip_enviro_stats_summary_file = open(os.path.join(save_path_start,trip_enviro_stats_summary_file_name),'a')
    initiate_trip_enviro_stats_summary_file(file=trip_enviro_stats_summary_file)

if generate_trip_cf_stats_summary_file is True:
    trip_cf_stats_summary_file = open(os.path.join(save_path_start,trip_cf_stats_summary_file_name),'a')
    initiate_trip_cf_stats_summary_file(file=trip_cf_stats_summary_file)

if generate_trip_cf_enviro_stats_summary_file is True:
    trip_cf_enviro_stats_summary_file = open(os.path.join(save_path_start,trip_cf_enviro_stats_summary_file_name),'a')
    initiate_trip_cf_enviro_stats_summary_file(file=trip_cf_enviro_stats_summary_file)

if generate_trip_cf_enviro_stats_demo_behav_summary_file is True:
    trip_cf_enviro_stats_demo_behav_summary_file = open(os.path.join(save_path_start,trip_cf_enviro_stats_demo_behav_summary_file_name),'a')
    initiate_trip_cf_enviro_stats_demo_behav_summary_file(file=trip_cf_enviro_stats_demo_behav_summary_file)

if generate_TS_summary_file is True:
    TS_summary_file = open(os.path.join(save_path_start, TS_summary_file_name), 'a')
    initiate_ts_summary_file(file=TS_summary_file)

if generate_CFS_summary_file is True:
    CFS_summary_file = open(os.path.join(save_path_start, CFS_summary_file_name), 'a')
    initiate_cfs_summary_file(file=CFS_summary_file)

if generate_phase1_trip_enviro_stats_summary_file is True:
    phase1_trip_enviro_stats_summary_file = open(os.path.join(save_path_start, phase1_trip_enviro_stats_summary_file_name), 'a')
    initiate_phase1_trip_enviro_stats_summary_file(file=phase1_trip_enviro_stats_summary_file)

"""
Iterative Analysis on each Trip File
"""
counter = 0
for iterator in range(len(trip_num)):

    counter += 1
    trip_num_this = trip_num[iterator][0]  # trip number for current iteration
    print "  {}: {}".format(counter,trip_num_this)

    save_path = save_path_start + '\\{}_Results'.format(trip_num_this)

    NDSfile = '{}_{}.csv'.format(nds_file_name_start,trip_num_this)
    STACfile = '{}_stac_{}.csv'.format(nds_file_name_start,trip_num_this)
    try:
        point_collection = import_wy_nds_stac(nds_file_name=NDSfile,
                                          nds_path=open_nds_data_path,stac_file_name=STACfile,
                                          stac_path=open_stac_data_path)
        stac_data_available = True
    except IOError:
        point_collection = import_wy_nds(nds_file_name=NDSfile,
                                          path=open_nds_data_path)
        stac_data_available = False
        print "    "+"*STAC Radar Data Not Available"

    cf_collections = point_collection.car_following_event_extraction(min_cf_time=min_cf_time,
                                                                    max_cf_dist=max_cf_dist,
                                                                    min_speed=min_speed)
    if moving_average_filter is True:
        # Conduct a moving average filter of relative velocity - different for collected and calculated data
        if stac_data_available == True:
            for collection in cf_collections:
                collection.moving_average(14)
        elif stac_data_available == False:
            for collection in cf_collections:
                collection.moving_average(14)

    # Determine if Manual Observations Template is Complete
    try:
        import_obs_file(filename='{}_C_output_complete.csv'.format(trip_num_this),path=completed_observations_path)
        manual_observations_complete = True
    except IOError:
        manual_observations_complete = False
        print "    " + "*Manual Video Observations Not Complete"

    ##### Single Trip Functions: #####
    if save_reduced_output_file_for_full_trip is True:
        initialize_save_path(save_path)
        reduced_file = open(os.path.join(save_path,'{}_timeseries.csv'.format(trip_num_this)),'w')
        point_collection.export_timeseries_file(file=reduced_file)
        del reduced_file

        print "    "+"Reduced File Generated"

    ##### Time Segmentation (TS)#####
    if TS_save_split_time_series_files is True:
        # Split complete NDS trip into trip segments based on user-defined time increment
        split_point_collections = point_collection.segmentation(split_length=60)

        initialize_save_path(save_path+'\\TS-Reduced Timeseries')
        for i in range(len(split_point_collections)):
            split_file = open(os.path.join(save_path+'\\TS-Reduced Timeseries','{}_{}_timeseries.csv'.format(trip_num_this,i+1)),'w')
            split_point_collections[i].export_timeseries_file(file=split_file)
            del split_file
        del split_point_collections

        print "    "+"TS: Reduced Files Generated"

    if TS_create_manual_video_observation_template is True:
        # Generate video observation template
        observation_file = open(os.path.join(save_path,'{}_C_output.csv'.format(trip_num_this)),'w')
        point_collection.video_observation_template(file=observation_file,trip_no=trip_num_this,split_length=60)
        del observation_file

        print "    "+"TS: Manual Observation Template Complete"

    if TS_generate_descriptive_statistics is True:
        # Generate statistics for complete trip
        initialize_save_path(save_path + '\\TS-Statistics')
        statistics_file = open(os.path.join(save_path+'\\TS-Statistics','{}_statistics.csv'.format(trip_num_this)),'w')
        point_collection.summary_statistics(file=statistics_file,trip_no=trip_num_this)
        # Generate statistics for each segment
        split_point_collections = point_collection.segmentation(split_length=60)
        for i in range(len(split_point_collections)):
            label = "{}_{}".format(trip_num_this, i + 1)
            statistics_file = open(os.path.join(save_path+'\\TS-Statistics','{}_{}_statistics.csv'.format(trip_num_this,i+1)),'w')
            split_point_collections[i].summary_statistics(file=statistics_file,trip_no=label)
            del label
            del statistics_file
        del split_point_collections

        print "    "+"TS: Summary Statistics Complete"

    if TS_generate_time_series_plots is True:
        # Generate plot for complete trip
        plot_fig = point_collection.timeseries_summary_plots(trip_no=trip_num_this)
        initialize_save_path(save_path+'\\TS-Plotting')
        plot_fig.savefig(os.path.join(save_path+'\\TS-Plotting', '{}_plot'.format(trip_num_this)))
        # Generate statistics for each segment
        split_point_collections = point_collection.segmentation(split_length=60)
        for i in range(len(split_point_collections)):
            label = "{}_{}".format(trip_num_this,i+1)
            plot_fig = split_point_collections[i].timeseries_summary_plots(trip_no=label)
            plot_fig.savefig(os.path.join(save_path + '\\TS-Plotting', '{}_{}_plot'.format(trip_num_this,i+1)))
            del plot_fig
            del label
        del split_point_collections

        print "    "+"TS: Time Series Plotting Complete"

    ##### Car-Following Segmentation (CFS)#####
    if CFS_generate_individual_cf_plots is True:
        initialize_save_path(save_path + '\\CFS-Plotting\\Car-Following')
        for i in range(len(cf_collections)):
            label = "{}_{}_t-X".format(trip_num_this, i + 1)
            plot_fig = cf_collections[i].plot_t_X(trip_no=label)
            plot_fig.savefig(os.path.join(save_path + '\\CFS-Plotting\\Car-Following', '{}_{}_t_X'.format(trip_num_this, i + 1)))
            label = "{}_{}_dV-X".format(trip_num_this, i + 1)
            plot_fig = cf_collections[i].plot_dV_X(trip_no=label)
            plot_fig.savefig(os.path.join(save_path + '\\CFS-Plotting\\Car-Following', '{}_{}_dV_X'.format(trip_num_this, i + 1)))
            label = "{}_{}_t-dV".format(trip_num_this, i + 1)
            plot_fig = cf_collections[i].plot_t_dV(trip_no=label)
            plot_fig.savefig(os.path.join(save_path + '\\CFS-Plotting\\Car-Following', '{}_{}_t_dV'.format(trip_num_this, i + 1)))
            del plot_fig
            del label

        print "    " + "CFS: Car-Following Plotting Complete"

    if CFS_generate_cf_plot is True:
        initialize_save_path(save_path + '\\CFS-Plotting\\Car-Following')
        for i in range(len(cf_collections)):
            label = "{}_{}".format(trip_num_this, i + 1)
            plot_fig = cf_collections[i].cf_summary_plot(trip_no=label)
            plot_fig.savefig(os.path.join(save_path + '\\CFS-Plotting\\Car-Following', '{}_{}_cf_plot'.format(trip_num_this, i + 1)))

    if CFS_generate_time_series_plots is True:
        initialize_save_path(save_path + '\\CFS-Plotting\\Time Series')
        for i in range(len(cf_collections)):
            label = "{}_{}".format(trip_num_this,i+1)
            plot_fig = cf_collections[i].timeseries_summary_plots(trip_no=label)
            plot_fig.savefig(os.path.join(save_path + '\\CFS-Plotting\\Time Series', '{}_{}_timeseries_plot'.format(trip_num_this,i+1)))
            del plot_fig
            del label

        print "    " + "CFS: Time Series Plotting Complete"

    if CFS_save_time_series_file is True:
        # Split complete NDS trip into trip segments based on user-defined time increment
        initialize_save_path(save_path+'\\CFS-Reduced Timeseries')
        for i in range(len(cf_collections)):
            cf_file = open(os.path.join(save_path+'\\CFS-Reduced Timeseries','{}_{}_timeseries.csv'.format(trip_num_this,i+1)),'w')
            cf_collections[i].export_timeseries_file(file=cf_file)
            del cf_file

        print "    "+"CFS: Reduced Files Generated"

    ##### Multi-Trip Functions: #####
    if generate_variable_exist_summary_file is True:
        append_to_var_exist_summary_file(point_collection=point_collection, file=var_exist_summary_file,
                                         trip_no=trip_num_this)

        print "    " + "Variable Exist Summary File Complete"

    if generate_trip_enviro_stats_summary_file is True:
        if manual_observations_complete is True:
            obs_list = import_obs_file(filename='{}_C_output_complete.csv'.format(trip_num_this),path=completed_observations_path)
            append_to_trip_enviro_stats_summary_file(point_collection=point_collection,file=trip_enviro_stats_summary_file,trip_no=trip_num_this,observations=obs_list)
        else:
            append_to_trip_enviro_stats_summary_file(point_collection=point_collection,file=trip_enviro_stats_summary_file,trip_no=trip_num_this)

        print "    " + "Trip Enviro-Stats Summary File Complete"

    if generate_trip_cf_stats_summary_file is True:
        driver_id = get_driver_id(data_dictionary_file_name=data_dictionary_file_name, path=survey_path,
                                  trip_no=trip_num_this)
        append_to_trip_cf_stats_summary_file(point_collection=point_collection,cf_collections=cf_collections,driver_id=driver_id,
                                       file=trip_cf_stats_summary_file,trip_no=trip_num_this,
                                       stac_data_available=stac_data_available)

        print "    " + "Trip Car-Following Summary File Complete"

    if generate_trip_cf_enviro_stats_summary_file is True:
        if manual_observations_complete is True:
            obs_list = import_obs_file(filename='{}_C_output_complete.csv'.format(trip_num_this),path=completed_observations_path)
            append_to_trip_cf_enviro_stats_summary_file(point_collection=point_collection,cf_collections=cf_collections,
                                                        file=trip_cf_enviro_stats_summary_file,trip_no=trip_num_this,
                                                        stac_data_available=stac_data_available,observations=obs_list)
        else:
            append_to_trip_cf_enviro_stats_summary_file(point_collection=point_collection,cf_collections=cf_collections,
                                                        file=trip_cf_enviro_stats_summary_file,trip_no=trip_num_this,
                                                        stac_data_available=stac_data_available)

        print "    " + "Trip CF-Enviro-Stats Summary File Complete"

    if generate_trip_cf_enviro_stats_demo_behav_summary_file is True:

        driver_id = get_driver_id(data_dictionary_file_name=data_dictionary_file_name,path=survey_path,trip_no=trip_num_this)

        if np.isnan(driver_id) != True:
            demographics_data = get_demographics_survey_data(file_name=demographics_survey_file_name,path=survey_path,
                                                            driver_id=driver_id)
            behavior_data = get_behavior_survey_data(file_name=behavior_survey_file_name,path=survey_path,
                                                            driver_id=driver_id)
        else:
            demographics_data = None
            behavior_data = None

        if manual_observations_complete is True:
            obs_list = import_obs_file(filename='{}_C_output_complete.csv'.format(trip_num_this),
                                       path=completed_observations_path)
            append_to_trip_cf_enviro_stats_demo_behav_summary_file(point_collection=point_collection,
                                                        cf_collections=cf_collections,
                                                        file=trip_cf_enviro_stats_demo_behav_summary_file,
                                                        trip_no=trip_num_this,
                                                        stac_data_available=stac_data_available,
                                                        observations=obs_list,demographics_data=demographics_data,
                                                                   behavior_data=behavior_data, driver_id=driver_id)
        else:
            append_to_trip_cf_enviro_stats_demo_behav_summary_file(point_collection=point_collection,
                                                        cf_collections=cf_collections,
                                                        file=trip_cf_enviro_stats_demo_behav_summary_file,
                                                        trip_no=trip_num_this,
                                                        stac_data_available=stac_data_available,
                                                        observations=None,demographics_data=demographics_data,
                                                                   behavior_data=behavior_data,driver_id=driver_id)

        print "    " + "Trip CF-Enviro-Stats-Demo-Behav Summary File Complete"

    if generate_TS_summary_file is True:
        if manual_observations_complete is True:
            obs_list = import_obs_file(filename='{}_C_output_complete.csv'.format(trip_num_this),path=completed_observations_path)
            append_to_ts_summary_file(point_collection=point_collection,file=TS_summary_file,trip_no=trip_num_this,observations=obs_list)
        else:
            append_to_ts_summary_file(point_collection=point_collection,file=TS_summary_file,trip_no=trip_num_this)

        print "    "+"TS Summary File Complete"

    if generate_CFS_summary_file is True:
        if manual_observations_complete is True:
            obs_list = import_obs_file(filename='{}_C_output_complete.csv'.format(trip_num_this),path=completed_observations_path)
            append_to_cfs_summary_file(cf_collections=cf_collections,file=CFS_summary_file,trip_no=trip_num_this,
                                       stac_data_available=stac_data_available,observations=obs_list)
        else:
            append_to_cfs_summary_file(cf_collections=cf_collections,file=CFS_summary_file,trip_no=trip_num_this,
                                       stac_data_available=stac_data_available)

        print "    "+"CFS Summary File Complete"

    if generate_phase1_trip_enviro_stats_summary_file is True:
        if manual_observations_complete is True:
            obs_list = import_obs_file(filename='{}_C_output_complete.csv'.format(trip_num_this),path=completed_observations_path)
            append_to_phase1_trip_enviro_stats_summary_file(point_collection=point_collection,file=phase1_trip_enviro_stats_summary_file,trip_no=trip_num_this,observations=obs_list)
        else:
            append_to_phase1_trip_enviro_stats_summary_file(point_collection=point_collection,file=phase1_trip_enviro_stats_summary_file,trip_no=trip_num_this)

        print "    " + "Phase 1 Trip Enviro-Stats Summary File Complete"


if generate_variable_exist_summary_file is True:
    var_exist_summary_file.close()
if generate_trip_enviro_stats_summary_file is True:
    trip_enviro_stats_summary_file.close()
if generate_trip_cf_stats_summary_file is True:
    trip_cf_stats_summary_file.close()
if generate_trip_cf_enviro_stats_summary_file is True:
    trip_cf_enviro_stats_summary_file.close()
if generate_trip_cf_enviro_stats_demo_behav_summary_file is True:
    trip_cf_enviro_stats_demo_behav_summary_file.close()
if generate_CFS_summary_file is True:
    CFS_summary_file.close()
if generate_TS_summary_file is True:
    TS_summary_file.close()