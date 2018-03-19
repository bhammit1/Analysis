import os
from tools.functions_basic import import_wy_nds_stac
from tools.functions_basic import import_wy_nds
from tools.functions_aggBins import create_aggregate_plots
from tools.functions_aggBins import reaggregate_bins
from scipy import stats

"""
Script to create aggregated bins from Wyoming NDS data - creates the aggregated bins for actual weather frameworks 
by aggregating trips and car-following events together. 

This is using the same data as the second (Dec 2017 Gipps Case Study) - the trip sets selected for the analysis 
were verified to have comparable car-following exposure.
"""
date = '2018-02-08_fixed'

open_global_input_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.01.22 WyNdsAggregatedBins\Inputs'
open_nds_data_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2017.11.15 GippsWeather\CalibrationInputs\NDS Data'
open_stac_data_path = open_nds_data_path
save_path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\07-Results\2018.01.22 WyNdsAggregatedBins'

nds_file_name_start = 'Event_ID'

data_dictionary_file_name = 'data_dictionary_casestudy.csv'

frameworks = ['Fog','Very Light Rain','Light Rain','Moderate Rain','Heavy Rain','Snow']
#frameworks = ['Fog']

##### Car-Following Settings #####
min_cf_time = 20  # s
max_cf_dist = 60  # m
min_speed = 1  # m/s

##### Moving Average Filter Settings for Radar Data #####
moving_average_filter = True

# Read in trip numbers from file
data_dictionary_matrix = list()
data_dictionary_file = open(os.path.join(open_global_input_path,data_dictionary_file_name),'r')
data_dictionary_file.next()
data_dictionary_file.next()
for line in data_dictionary_file:
    data_dictionary_matrix.append(line.strip().split(','))
    # trip_set_no, clear_trip_no,adverse_trip_no,condition,driver_id
data_dictionary_file.close()
del data_dictionary_file

framework_trip_no_dict = {}
for i in range(len(frameworks)):
    # Initialize framework dictionary
    framework_trip_no_dict['{}_a'.format(i+1)] = list()
    framework_trip_no_dict['{}_b'.format(i+1)] = list()
for i in range(len(data_dictionary_matrix)):
    for j in range(len(frameworks)):
        if data_dictionary_matrix[i][3] == frameworks[j]:
            framework_trip_no_dict['{}_a'.format(j+1)].append(data_dictionary_matrix[i][2])
            framework_trip_no_dict['{}_b'.format(j+1)].append(data_dictionary_matrix[i][1])

framework_bins_dict = {}

for framework,trip_nos in framework_trip_no_dict.iteritems():

    print "Framework: {} with {} trips".format(framework, len(trip_nos))

    fw_cf_collections = list()

    for trip_no in trip_nos:

        save_path_this = save_path + '\\{}_Results'.format(trip_no)
        NDSfile = '{}_{}.csv'.format(nds_file_name_start,trip_no)
        STACfile = '{}_stac_{}.csv'.format(nds_file_name_start,trip_no)
        try:
            point_collection = import_wy_nds_stac(nds_file_name=NDSfile,nds_path=open_nds_data_path,
                                                  stac_file_name=STACfile,stac_path=open_stac_data_path)
            stac_data_available = True
        except IOError:
            point_collection = import_wy_nds(nds_file_name=NDSfile,path=open_nds_data_path)
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

        # Append to FW_CF_Collection:
        for c in cf_collections:
            fw_cf_collections.append(c)

    save_file = open(os.path.join(save_path,'{}_{}_bin.csv'.format(date,framework)),'w')
    binned_data = create_aggregate_plots(plot_content='acceleration',plot_evaluation='mean',cf_collections=fw_cf_collections,file=save_file)

    reaggregated_bined_data = reaggregate_bins(binned_data)

    framework_bins_dict['{}'.format(framework)] = reaggregated_bined_data

##### T-Test Analysis #####

print "T-Test Analysis"

# Adverse vs. Matching Clear
target = open(os.path.join(save_path,'{}_T_adv_clear.csv'.format(date)),'w')
target.write('Adverse vs. Clear')
target.write('\n')
adv_clear = list()
for i in range(len(frameworks)):
    adv_clear.append(['{}_a'.format(i+1),'{}_b'.format(i+1)])

index_dict = {}
rows = len(framework_bins_dict['1_a'])  # all are the same, just using 1a for sample
cols = len(framework_bins_dict['1_a'][0])
for i in range(rows):
    for j in range(cols):
        index_dict['{},{}'.format(i,j)] = {}

for i in range(len(adv_clear)):
    target.write('FW: {}'.format(adv_clear[i]))
    target.write('\n')
    fw1 = framework_bins_dict['{}'.format(adv_clear[i][0])]
    fw2 = framework_bins_dict['{}'.format(adv_clear[i][1])]
    for j in range(len(fw1)):
        for k in range(len(fw1[j])):
            t_test = stats.ttest_ind(fw1[j][k],fw2[j][k],nan_policy='omit')
            index_dict['{},{}'.format(j, k)]['{}-{}'.format(adv_clear[i][0], adv_clear[i][1])] = t_test[1]
            if isinstance(t_test[1],float):
                target.write('{},{},{}'.format(j,k,t_test[1]))
                target.write('\n')

target.close()
del target

target = open(os.path.join(save_path,'{}_T_adv_clear2.csv'.format(date)),'w')
target.write('Adverse vs. Clear')
target.write('\n')
target.write('row,col,')
for key, p_val in sorted(index_dict['1,1'].iteritems()):
    target.write('{},'.format(key))
target.write('\n')
for index, dicts in sorted(index_dict.iteritems()):
    target.write('{},'.format(index))
    for key, p_val in sorted(dicts.iteritems()):
        target.write('{},'.format(p_val))
    target.write('\n')
target.close()
del target, index_dict


# Adverse vs. Adverse (A)
target = open(os.path.join(save_path,'{}_T_adv_adv.csv'.format(date)),'w')
target.write('Adverse vs. Adverse')
target.write('\n')
adv_adv = list()
for i in range(len(frameworks)):
    for j in range(len(frameworks)):
        adv_adv.append(['{}_a'.format(i+1),'{}_a'.format(j+1)])

index_dict = {}
rows = len(framework_bins_dict['1_a'])  # all are the same, just using 1a for sample
cols = len(framework_bins_dict['1_a'][0])
for i in range(rows):
    for j in range(cols):
        index_dict['{},{}'.format(i,j)] = {}

for i in range(len(adv_adv)):
    target.write('FW: {}'.format(adv_adv[i]))
    target.write('\n')
    fw1 = framework_bins_dict['{}'.format(adv_adv[i][0])]
    fw2 = framework_bins_dict['{}'.format(adv_adv[i][1])]
    for j in range(len(fw1)):
        for k in range(len(fw1[j])):
            t_test = stats.ttest_ind(fw1[j][k],fw2[j][k],nan_policy='omit')
            index_dict['{},{}'.format(j, k)]['{}-{}'.format(adv_adv[i][0], adv_adv[i][1])] = t_test[1]
            if isinstance(t_test[1],float):
                target.write('{},{},{}'.format(j,k,t_test[1]))
                target.write('\n')
target.close()
del target

target = open(os.path.join(save_path,'{}_T_adv_adv2.csv'.format(date)),'w')
target.write('Adverse vs. Adverse')
target.write('\n')
target.write('row,col,')
for key, p_val in sorted(index_dict['1,1'].iteritems()):
    target.write('{},'.format(key))
target.write('\n')
for index, dicts in sorted(index_dict.iteritems()):
    target.write('{},'.format(index))
    for key, p_val in sorted(dicts.iteritems()):
        target.write('{},'.format(p_val))
    target.write('\n')
target.close()
del target, index_dict

# Clear vs. Clear (B)
target = open(os.path.join(save_path,'{}_T_clear_clear.csv'.format(date)),'w')
target.write('Clear vs. Clear')
target.write('\n')
clear_clear = list()
for i in range(len(frameworks)):
    for j in range(len(frameworks)):
        clear_clear.append(['{}_b'.format(i+1),'{}_b'.format(j+1)])

index_dict = {}
rows = len(framework_bins_dict['1_a'])  # all are the same, just using 1a for sample
cols = len(framework_bins_dict['1_a'][0])
for i in range(rows):
    for j in range(cols):
        index_dict['{},{}'.format(i,j)] = {}

for i in range(len(clear_clear)):
    target.write('FW: {}'.format(clear_clear[i]))
    target.write('\n')
    fw1 = framework_bins_dict['{}'.format(clear_clear[i][0])]
    fw2 = framework_bins_dict['{}'.format(clear_clear[i][1])]
    for j in range(len(fw1)):
        for k in range(len(fw1[j])):
            t_test = stats.ttest_ind(fw1[j][k],fw2[j][k],nan_policy='omit')
            index_dict['{},{}'.format(j, k)]['{}-{}'.format(clear_clear[i][0], clear_clear[i][1])] = t_test[1]
            if isinstance(t_test[1],float):
                target.write('{},{},{}'.format(j,k,t_test[1]))
                target.write('\n')

target.close()
del target

target = open(os.path.join(save_path,'{}_T_clear_clear2.csv'.format(date)),'w')
target.write('Clear vs. Clear')
target.write('\n')
target.write('row,col,')
for key, p_val in sorted(index_dict['1,1'].iteritems()):
    target.write('{},'.format(key))
target.write('\n')
for index, dicts in sorted(index_dict.iteritems()):
    target.write('{},'.format(index))
    for key, p_val in sorted(dicts.iteritems()):
        target.write('{},'.format(p_val))
    target.write('\n')
target.close()
del target, index_dict