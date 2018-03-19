import os

# IDM
"""
path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\03-Python Script Files\2018.02.15 All FHWA Trips Calibration Main Scripts\idm_inputs'

all_trip_ids_filename = 'idm_ALL_trip_ids.csv'

all_trip_ids_file = open(os.path.join(path,all_trip_ids_filename),'r')
all_trip_ids = list()
for line in all_trip_ids_file:
    all_trip_ids.append(line.strip().split(','))

split_index = [45,90,135,180,225,270,315,360,404,448,493,538,583,628,673,718,763,808,852,896,1008,1120,1232,1344]
name_index = [39,39,39,39,39,39,39,39,39,41,41,41,41,41,41,41,41,41,41,41,'rach_fhwa','rach_fhwa','rach_fhwa','rach_fhwa']
counter_index = [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10,1,2,3,4]

index = 0
for i in range(len(split_index)):
    target = open(os.path.join(path, 'idm_{}_trip_ids_{}.csv'.format(name_index[i], counter_index[i])), 'w')
    while index < split_index[i]:
        target.write('{}'.format(all_trip_ids[index][0]))
        target.write('\n')
        index += 1
"""
# Newell
"""
path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\03-Python Script Files\2018.02.15 All FHWA Trips Calibration Main Scripts\newell_inputs'

all_trip_ids_filename = 'newell_ALL_trip_ids.csv'

all_trip_ids_file = open(os.path.join(path,all_trip_ids_filename),'r')
all_trip_ids = list()
for line in all_trip_ids_file:
    all_trip_ids.append(line.strip().split(','))

split_index = [68,136,203,270,337,404,471,538,605,672,740,808,875,942,1009,1076,1143,1210,1277,1344]
name_index = [35,35,35,35,35,35,35,35,35,35,37,37,37,37,37,37,37,37,37,37]
counter_index = [1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9,10]

index = 0
for i in range(len(split_index)):
    target = open(os.path.join(path, 'newell_{}_trip_ids_{}.csv'.format(name_index[i], counter_index[i])), 'w')
    while index < split_index[i]:
        target.write('{}'.format(all_trip_ids[index][0]))
        target.write('\n')
        index += 1
"""
# W99
"""
path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\03-Python Script Files\2018.02.15 All FHWA Trips Calibration Main Scripts\w99_inputs'

all_trip_ids_filename = 'w99_ALL_trip_ids.csv'

all_trip_ids_file = open(os.path.join(path,all_trip_ids_filename),'r')
all_trip_ids = list()
for line in all_trip_ids_file:
    all_trip_ids.append(line.strip().split(','))

split_index = [168,336,504,672,840,1008,1176,1344]
name_index = ['britt_lenovo','britt_lenovo','britt_lenovo','britt_lenovo','rach_samsung','rach_samsung','rach_samsung','rach_samsung']
counter_index = [1,2,3,4,1,2,3,4]

index = 0
for i in range(len(split_index)):
    target = open(os.path.join(path, 'w99_{}_trip_ids_{}.csv'.format(name_index[i], counter_index[i])), 'w')
    while index < split_index[i]:
        target.write('{}'.format(all_trip_ids[index][0]))
        target.write('\n')
        index += 1
"""
# Gipps

path = r'C:\Users\britton.hammit\Dropbox\00-Education\research\03-NDS_and_RID\WY_SHRP2_NDS\Python_Analysis\03-Python Script Files\2018.02.15 All FHWA Trips Calibration Main Scripts\gipps_inputs'

all_trip_ids_filename = 'gipps_ALL_trip_ids.csv'

all_trip_ids_file = open(os.path.join(path,all_trip_ids_filename),'r')
all_trip_ids = list()
for line in all_trip_ids_file:
    all_trip_ids.append(line.strip().split(','))

split_index = [269,538,807,1076,1344]
name_index = ['britt_fhwa','britt_fhwa','britt_fhwa','britt_fhwa','britt_fhwa']
counter_index = [1,2,3,4,5]

index = 0
for i in range(len(split_index)):
    target = open(os.path.join(path, 'gipps_{}_trip_ids_{}.csv'.format(name_index[i], counter_index[i])), 'w')
    while index < split_index[i]:
        target.write('{}'.format(all_trip_ids[index][0]))
        target.write('\n')
        index += 1
