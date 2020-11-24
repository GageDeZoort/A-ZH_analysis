import os
import json
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--year', default='')
args = parser.parse_args()

eras = {'2016':'Summer16', '2017':'Fall17', '2018':'Autumn18'}
era = eras[args.year]

# open sample file
infile = open("data_{0}.csv".format(args.year))
data_samples = [f.strip() for f in infile]
print(data_samples)

# loop over all samples
outfile = open("all_{0}_data.yaml".format(args.year), "w+")
outfile.write("data_{0}:\n".format(args.year))
for loc in data_samples:
    name = loc.split('/')[1] + '_' + loc.split('/')[2].split('-')[0]

    # get sample list
    query = '"dataset={0}"'.format(loc)
    command = 'dasgoclient --query={0}'.format(query)
    print('...executing:\n', command)
    sample = subprocess.check_output(command, shell=True).decode().split('\n')[0]

    # list all files in sample
    print('...sample:', sample)
    
    query = '"file dataset={0}"'.format(sample)
    command = 'dasgoclient --query={0}'.format(query)
    try: 
        sample_files = subprocess.check_output(command, shell=True)
    except: 
        continue
    
    sample_files = sample_files.split('\n')[:-1]
    for sample_file in sample_files:
        outfile.write("  - " + "root://cmsxrootd.fnal.gov/" + 
                      sample_file.strip('\n') + "\n")

outfile.close()


