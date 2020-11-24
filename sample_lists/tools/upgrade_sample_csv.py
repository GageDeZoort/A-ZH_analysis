import sys
import os
import subprocess
from subprocess import PIPE

infile = sys.argv[1]
eras = {2016:'Summer16', 2017:'Fall17', 2018:'Autumn18'} 
year = int(infile.split('_')[3].split('.')[0])
era  = eras[year]
print("...running on {0} {1}".format(year, era))

outfile = infile.split("_v5")[0] + '_v7.csv'
outlines = []
for line in open(infile, 'r').readlines():
    sample_info = line.split(',')
    dataset_name = sample_info[6].split('/')[1]
    query = '\"dataset=/{0:s}/*NanoAODv6*/NANOAODSIM\"'.format(dataset_name)
    command = 'dasgoclient --query={0:s}'.format(query)
    print("\n***** {0} ***** ".format(sample_info[0]))
    print("...query: \n{0}".format(query))

    results = subprocess.check_output(command, shell=True).decode().split('\n')[:-1]
    
    if (len(results) < 1): 
        print("ERROR: unexpected results")
        continue
    else:
        print("...results:")
        for result in results:
            if era in result:
                print("{0}".format(result))
                new_dataset = result.strip()
                name = sample_info[0]
                outlines.append("{0:s}, {1:s}, {2:s}, {3:s}, {4:s}, {5:s}, {6:s}\n"
                                .format(name, sample_info[1], sample_info[2], sample_info[3], 
                                        sample_info[4], sample_info[5], new_dataset))

open(outFileName, 'w').writelines(outLines)
