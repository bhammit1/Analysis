import timeit
from copy import deepcopy
from tools.functions_basic import import_trip_ids
from tools.functions_basic import check_files_exist
from tools.functions_basic import generate_cf_collections_nds
from tools.functions_idm import *
from tools.functions_newell import *
from tools.functions_gipps import *
from tools.functions_w99 import *


# Save Scheme:
# ----------------------------------------------------------------------------------------
date = '2018-03-12_lit'  # Naming for output files - Cannot be with dots - must be with dashes

# Identify Paths:
# ----------------------------------------------------------------------------------------
trip_id_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\04-Dissertation\Rachel_Britton\03-Projects\2018.02.07 ITSA NDS Calibration Procedures\2018.03.08 Case Study\Calibration\Inputs'
nds_data_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 a.Nds_Stac_Files'
stac_data_input_path = nds_data_input_path
driver_survey_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 GippsCalibration\Global_Inputs'
save_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\04-Dissertation\Rachel_Britton\03-Projects\2018.02.07 ITSA NDS Calibration Procedures\2018.03.08 Case Study\Validation'
nds_file_name_start = 'Event_ID'

# Car-Following Event Requirements (only applicable to NDS and Synthetic)
# ----------------------------------------------------------------------------------------
min_cf_time, max_cf_dist, min_speed = 20, 60, 1  # s, m, m/s

# Input files names and paths:
# ----------------------------------------------------------------------------------------
trip_nos_filename_list = [['itsa_calibration_lowspeed.csv','itsa_validation_lowspeed.csv'],
                          ['itsa_calibration_highspeed.csv','itsa_validation_highspeed.csv'],
                          ['itsa_calibration_all.csv','itsa_validation_all.csv']]
"""
# Calibrated Values
gipps_individuals = [[1.1,35.6,1.8,-1.9,-1.7,0.2],
                     [0.4,40.0,0.6,-2.1,-2.1,4.3],
                     [1.1,35.3,1.6,-1.3,-1.2,0.6]]
idm_individuals = [[1.8,0.3,100,35.5,0.7,3.5],
                   [0.3,1.1,78,35.8,0.6,2.3],
                   [1.7,0.4,87,35.9,0.6,4.2]]
w99_individuals = [[4.3,0.6,13.3,-27.0,0.0,0.9,0.5,4.0,0.1,0.1,35.7,850],
                   [7.9,0.5,14.4,-26.3,0.0,0.0,0.6,2.6,0.2,0.2,38.8,980],
                   [5.7,0.6,13.3,-26.7,0.0,0.0,0.8,2.3,0.9,0.1,35.4,418]]
newell_individuals = [[0.8,3.6],
                      [0.5,9.5],
                      [0.7,6.6]]
"""
# Literature Values
gipps_individuals = [[0.7,35.0,2.0,-3.0,-3.5,1.0],
                     [0.7,35.0,2.0,-3.0,-3.5,1.0],
                     [0.7,35.0,2.0,-3.0,-3.5,1.0]]  # Average len of vehicle is 4.5 m (Olstam, Tapani 2004 - 1.0 & Gipps original formulation)
idm_individuals = [[1.4,2.0,4,35.0,1.5,2.0],
                   [1.4,2.0,4,35.0,1.5,2.0],
                   [1.4,2.0,4,35.0,1.5,2.0]]  # Get source from Rachel
w99_individuals = [[1.5,1.3,4.0,-12.0,-0.3,0.4,6.0,0.3,2.0,1.5,35.0,1],
                   [1.5,1.3,4.0,-12.0,-0.3,0.4,6.0,0.3,2.0,1.5,35.0,1],
                   [1.5,1.3,4.0,-12.0,-0.3,0.4,6.0,0.3,2.0,1.5,35.0,1]]  # Using the 35.0m/s as speed for all models
newell_individuals = [[1.0,0.4],
                      [1.0,0.4],
                      [1.0,0.4]]  # From Punzo, Simonelli 2005 - average calibrated values

summary_file = open(os.path.join(save_path,'{}_itsA_scores.csv'.format(date)),'w')
summary_file.write('ITSA Calibration and Validation Score Summary File')
summary_file.write('\n')
summary_file.write(',Calibration,,,,Validation')
summary_file.write('\n')
summary_file.write('trip_filename,gipps_calib,idm_calib,newell_calib,w99_calib,gipps_vaid,idm_vaid,newell_vaid,w99_vaid')
summary_file.write('\n')



# Iterate for each input file:
file_counter = 0
for calib_valid_filenames in trip_nos_filename_list:

    summary_file.write('{},'.format(calib_valid_filenames[0].replace('itsa_calibration_','').replace('.csv','')))
    for trip_nos_filename in calib_valid_filenames:
        print ''
        print trip_nos_filename
        print ''
        # Read in trip numbers from file
        trip_no_list = import_trip_ids(trip_id_input_path, trip_nos_filename)

        # Check validity of each trip number before starting analysis
        check_files_exist(trip_no_list, nds_data_input_path, nds_file_name_start)

        print "---------------------------------------------------------------------------"

        # Collect all relevant car-following collections.
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

        # Calculate Calibration/Validation Scores

        # Gipps
        gipps_indiv = deepcopy(gipps_individuals[file_counter])
        for i in range(len(gipps_indiv)):
            gipps_indiv[i] = gipps_indiv[i] * 10.
        gipps_score = evaluate_gipps_GA(individual=gipps_indiv, cf_collections=cf_collections_all)[0]
        summary_file.write('{},'.format(gipps_score))
        print "Gipps Results: {}".format(gipps_score)
        print "----------------------------------------------"

        # IDM
        idm_indiv = deepcopy(idm_individuals[file_counter])
        for i in range(len(idm_indiv)):
            if i != 2:
                idm_indiv[i] = idm_indiv[i] * 10.
        idm_score = evaluate_idm_GA(individual=idm_indiv, cf_collections=cf_collections_all)[0]
        summary_file.write('{},'.format(idm_score))
        print "IDM Results: {}".format(idm_score)
        print "----------------------------------------------"

        # Newell
        newell_indiv = deepcopy(newell_individuals[file_counter])
        for i in range(len(newell_indiv)):
            newell_indiv[i] = newell_indiv[i] * 10.
        newell_score = evaluate_newell_GA(individual=newell_indiv, cf_collections=cf_collections_all)[0]
        summary_file.write('{},'.format(newell_score))
        print "Newell Results: {}".format(newell_score)
        print "----------------------------------------------"

        # W99
        w99_indiv = deepcopy(w99_individuals[file_counter])
        for i in range(len(w99_indiv)):
            if i != 11:
                w99_indiv[i] = w99_indiv[i] * 10.
        w99_score = evaluate_w99_GA(individual=w99_indiv, cf_collections=cf_collections_all)[0]
        summary_file.write('{},'.format(w99_score))
        print "W99 Results: {}".format(w99_score)
        print "----------------------------------------------"


    file_counter += 1
    print ""
    print ""
    summary_file.write('\n')

