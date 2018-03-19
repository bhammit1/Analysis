import os
import timeit
from tools.functions_basic import import_trip_ids
from tools.functions_basic import check_files_exist
from tools.functions_basic import generate_cf_collections_nds
from tools.functions_newell import run_newell_GA


# Save Scheme:
# ----------------------------------------------------------------------------------------
date = '2018-03-08_newellTEST'  # Naming for output files - Cannot be with dots - must be with dashes

# Identify Paths:
# ----------------------------------------------------------------------------------------
trip_id_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\04-Dissertation\Rachel_Britton\03-Projects\2018.02.07 ITSA NDS Calibration Procedures\2018.03.08 Case Study\Calibration\Inputs'
nds_data_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 a.Nds_Stac_Files'
stac_data_input_path = nds_data_input_path
driver_survey_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 newellCalibration\Global_Inputs'
save_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\04-Dissertation\Rachel_Britton\03-Projects\2018.02.07 ITSA NDS Calibration Procedures\2018.03.08 Case Study\Calibration\Results'

# Identify Input File Names:
# ----------------------------------------------------------------------------------------
trip_nos_filename_list = ['itsa_calibration_lowspeed.csv','itsa_calibration_highspeed.csv','itsa_calibration_all.csv']
behavior_survey_file_name = 'DrivingBehaviorQuestionnaire_UW.csv'
demographics_survey_file_name = 'DriverDemographicQuestionnaire_UW.csv'
data_dictionary_file_name = 'RainClear_Events_UW_A.csv'

print "---------------------------------------------------------------------------"

# GA Inputs - Identified from sensitivity analysis!
# ----------------------------------------------------------------------------------------
cxpb, mutpb, m_indpb, ngen, npop = 0.4, 0.2, 0.3, 50, 500

# Car-Following Event Requirements ** MUST MATCH MATCHING CRITERIA!
# ----------------------------------------------------------------------------------------
min_cf_time, max_cf_dist, min_speed = 20, 60, 1  # s, m, m/s

# Input Trip Numbers
# ----------------------------------------------------------------------------------------
nds_file_name_start = 'Event_ID'

# Iterate for each input file:
for trip_nos_filename in trip_nos_filename_list:
    # Read in trip numbers from file
    trip_no_list = import_trip_ids(trip_id_input_path, trip_nos_filename)

    # Check validity of each trip number before starting analysis
    check_files_exist(trip_no_list, nds_data_input_path, nds_file_name_start)

    print "---------------------------------------------------------------------------"

    cf_collections_all = list()
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

        print "   " + "No Car-Following Events: {}".format(len(cf_collections))

        for i in range(len(cf_collections)):
            cf_collections_all.append(cf_collections[i])

    print ""
    print "TOTAL No Car-Following Events: {}".format(len(cf_collections_all))
    print ""

    # Calibrate
    # ----------------------------------------------------------------------------------------
    calib_start_time = timeit.default_timer()
    logfile = open(os.path.join(save_path, "{}_{}_logfile.csv".format(date, trip_nos_filename)), 'w')
    calib_score, calib_best_indiv = run_newell_GA(cf_collections=cf_collections_all, cxpb=cxpb, mutpb=mutpb,
                                          m_indpb=m_indpb, ngen=ngen, npop=npop,
                                          logfile=logfile)
    logfile.close()

    total_time = timeit.default_timer() - start_time

    print "   " + "Total Time: {:4.3f} sec".format(total_time)
    print "---------------------------------------------------------------------------"

    print ""