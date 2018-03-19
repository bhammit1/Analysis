import os
import timeit
import numpy as np
from tools.functions_basic import import_trip_ids
from tools.functions_basic import check_files_exist
from tools.functions_basic import generate_cf_collections_nds
from tools.functions_basic import get_driver_id
from tools.functions_basic import get_demographics_survey_data
from tools.functions_basic import get_behavior_survey_data
from tools.functions_w99 import run_w99_GA
from tools.functions_w99 import initiate_201802Calib_w99_summary_file
from tools.functions_w99 import append_to_201802Calib_w99_summary_file


# Save Scheme:
# ----------------------------------------------------------------------------------------
date = '2018-02-15_w99'  # Naming for output files - Cannot be with dots - must be with dashes

# Identify Paths:
# ----------------------------------------------------------------------------------------
trip_id_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 W99Calibration\Global_Inputs'
nds_data_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 a.Nds_Stac_Files'
stac_data_input_path = nds_data_input_path
driver_survey_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 W99Calibration\Global_Inputs'
save_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 W99Calibration\Results'

# Identify Input File Names:
# ----------------------------------------------------------------------------------------
trip_no_for_processing = 'trip_id_directory_test2.csv'
behavior_survey_file_name = 'DrivingBehaviorQuestionnaire_UW.csv'
demographics_survey_file_name = 'DriverDemographicQuestionnaire_UW.csv'
data_dictionary_file_name = 'RainClear_Events_UW_A.csv'

print "---------------------------------------------------------------------------"

# GA Inputs - Identified from sensitivity analysis!
# ----------------------------------------------------------------------------------------
cxpb, mutpb, m_indpb, ngen, npop = 0.7, 0.4, 0.6, 60, 800

# Car-Following Event Requirements ** MUST MATCH MATCHING CRITERIA!
# ----------------------------------------------------------------------------------------
min_cf_time, max_cf_dist, min_speed = 20, 60, 1  # s, m, m/s

# Input Trip Numbers
# ----------------------------------------------------------------------------------------
nds_file_name_start = 'Event_ID'
# Read in trip numbers from file
trip_no_list = import_trip_ids(trip_id_input_path, trip_no_for_processing)

# Check validity of each trip number before starting analysis
check_files_exist(trip_no_list, nds_data_input_path, nds_file_name_start)

# Initiate Summary File
# ----------------------------------------------------------------------------------------
summary_file = open(os.path.join(save_path, '{}_summary_{}'.format(date,trip_no_for_processing)), 'a')
initiate_201802Calib_w99_summary_file(summary_file)
summary_file.close()  # Only open file when writing!

print "---------------------------------------------------------------------------"

for iterator in range(len(trip_no_list)):

    start_time = timeit.default_timer()

    trip_no_this = trip_no_list[iterator][0]  # trip number for current iteration
    print "{}:  {}".format(iterator + 1, trip_no_this)

    point_collection, cf_collections, stac_data_available = generate_cf_collections_nds(
                                                                      nds_file_name_start=nds_file_name_start,
                                                                      trip_no=trip_no_this,
                                                                      nds_path=nds_data_input_path,
                                                                      stac_path=stac_data_input_path,
                                                                      min_cf_time=min_cf_time, max_cf_dist=max_cf_dist,
                                                                      min_speed=min_speed)

    # Initialize Trip Path & Plot CF Events
    # ----------------------------------------------------------------------------------------
    print "   " + "No Car-Following Events: {}".format(len(cf_collections))

    # Calibrate
    # ----------------------------------------------------------------------------------------
    calib_start_time = timeit.default_timer()
    logfile = open(os.path.join(save_path, "{}_{}_logfile.csv".format(date, trip_no_this)), 'w')
    calib_score, calib_best_indiv = run_w99_GA(cf_collections=cf_collections, cxpb=cxpb, mutpb=mutpb,
                                          m_indpb=m_indpb, ngen=ngen, npop=npop,
                                          logfile=logfile)
    logfile.close()
    calib_time = timeit.default_timer() - calib_start_time

    print "   " + "   " + "Calib Time: {:4.3f} sec".format(calib_time)

    # Collect Driver Data
    # ----------------------------------------------------------------------------------------
    driver_id = get_driver_id(data_dictionary_file_name=data_dictionary_file_name, path=driver_survey_input_path,
                              trip_no=trip_no_this)

    if np.isnan(driver_id) != True:
        demographics_data = get_demographics_survey_data(file_name=demographics_survey_file_name, path=driver_survey_input_path,
                                                         driver_id=driver_id)
        behavior_data = get_behavior_survey_data(file_name=behavior_survey_file_name, path=driver_survey_input_path,
                                                 driver_id=driver_id)
    else:
        demographics_data = None
        behavior_data = None

    total_time = timeit.default_timer() - start_time

    print "   " + "Total Time: {:4.3f} sec".format(total_time)
    print "---------------------------------------------------------------------------"

    # Append to Summary File
    # ----------------------------------------------------------------------------------------
    summary_file = open(os.path.join(save_path, '{}_summary.csv'.format(date)), 'a')
    append_to_201802Calib_w99_summary_file(file=summary_file,
                                              trip_no=trip_no_this,
                                              driver_id=driver_id,
                                              point_collection=point_collection,
                                              cf_collections=cf_collections,
                                              stac_data_available=stac_data_available,
                                              demographics_data=demographics_data,
                                              behavior_data=behavior_data,
                                              calib_time=calib_time,
                                              calib_score=calib_score,
                                              calib_best_indiv=calib_best_indiv,
                                              total_time=total_time)
    summary_file.close()