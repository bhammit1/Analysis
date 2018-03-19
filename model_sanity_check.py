from tools.functions_basic import generate_cf_collections_nds
from tools.functions_idm import *
from tools.functions_newell import *
from tools.functions_gipps import *
from tools.functions_w99 import *

#save_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\04-Dissertation\Rachel_Britton\03-Projects\2018.02 Model Calibration\sanity_checks'

# Car-Following Event Requirements (only applicable to NDS and Synthetic)
# ----------------------------------------------------------------------------------------
min_cf_time, max_cf_dist, min_speed = 20, 60, 1  # s, m, m/s

# Input files names and paths:
# ----------------------------------------------------------------------------------------
Input_WyNDS = True
trip_no = 152219700
nds_file_name_start = 'Event_ID'
open_nds_data_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.02.15 a.Nds_Stac_Files'
open_stac_data_path = open_nds_data_path

point_collection,cf_collections,stac_data_available = generate_cf_collections_nds(nds_file_name_start=nds_file_name_start,
                                                                      trip_no=trip_no, nds_path=open_nds_data_path,
                                                                      stac_path=open_stac_data_path,
                                                                      min_cf_time=min_cf_time, max_cf_dist=max_cf_dist,
                                                                      min_speed=min_speed)


"""
for i in range(len(cf_collections)):
    cf_file = open(
        os.path.join(save_path, '{}_{}_timeseries.csv'.format(trip_no, i + 1)), 'w')
    cf_collections[i].export_timeseries_file(file=cf_file)
    del cf_file

print "    " + "CFS: Reduced Files Generated"
"""
# Gipps
individual = [1.9,34.3,1.6,-1.4,-1.2,0.2]
for i in range(len(individual)):
    individual[i] = individual[i]*10

# W99
"""
individual = [1.5,1.3,4.0,-12.,-0.25,0.35,6.,0.25,2.,1.5,28.3,1]
for i in range(len(individual)):
    individual[i] = individual[i]*10
"""
# Newell
"""
for i in range(len(individual)):
    individual[i] = individual[i]*10
"""
# IDM
"""
for i in range(len(individual)):
    if i != 2:
        individual[i] = individual[i]*10
"""

print evaluate_gipps_GA(individual=individual,cf_collections=cf_collections)

