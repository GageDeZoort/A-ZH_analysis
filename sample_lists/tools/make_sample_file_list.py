import os
import json
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default='AToZhToLLTauTau')
parser.add_argument('-y', '--year', default='')
args = parser.parse_args()

eras = {'2016':'Summer16', '2017':'Fall17', '2018':'Autumn18'}
era = eras[args.year]

# list all samples
query = '"dataset=/{0}*/*NanoAODv7*/NANOAODSIM"'.format(args.name)
command = 'dasgoclient --query={0}'.format(query)
print('...executing:\n', command)
samples = subprocess.check_output(command, shell=True).decode().split('\n')

# list all files in sample
print('...samples:')
outfile = open("MC_{1}/{0}_{1}_samples.txt".format(args.name, args.year), "w+")
for sample in samples:
    #print(" {0}".format(sample))
    query = '"file dataset={0}"'.format(sample)
    command = 'dasgoclient --query={0}'.format(query)
    try: sample_files = subprocess.check_output(command, shell=True)
    except: continue
    sample_files = sample_files.split('\n')[:-1]
    
    for sample_file in sample_files:
        if ((not sample_file) or (era not in sample_file)): continue
        command = 'edmFileUtil -d {0}'.format(sample_file)
        sample_file_loc = subprocess.check_output(command, shell=True)    
        print(sample_file_loc)
        outfile.write("'" + sample_file_loc.strip('\n') + "',\n")

outfile.close()


