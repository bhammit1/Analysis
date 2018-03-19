from os import listdir
from os.path import isfile, join

path = r'D:\All_trip_files'

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]



for i in range(len(onlyfiles)):
    onlyfiles[i] = onlyfiles[i].replace('Event_ID_','')
    onlyfiles[i] = onlyfiles[i].replace('.csv','')

print len(onlyfiles)

target = open(join(path,'0-trip_id_directory.csv'),'w')

for i in range(len(onlyfiles)):
    target.write('{}'.format(onlyfiles[i]))
    if i != len(onlyfiles)-1:
        target.write('\n')

target.close()