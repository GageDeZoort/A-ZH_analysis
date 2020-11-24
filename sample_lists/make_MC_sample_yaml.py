import os
import json
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default='all')
parser.add_argument('-y', '--year', default='')
parser.add_argument('-o', '--outdir', default='samples_all')
args = parser.parse_args()

eras = {'2016':'Summer16', '2017':'Fall17', '2018':'Autumn18'}
era = eras[args.year]

# open sample file
indir = "dataset_lists_MC/"
infile = open(indir+"MCsamples_{0}_paper.csv".format(args.year))
sample_dict = {line.split(',')[0]: line.split(',')[6].strip('\n') for line in infile}

# loop over all samples
outdir = args.outdir
outfile = open(outdir+"{0}_{1}_samples.yaml".format(args.name, args.year), "w+")
for name, loc in sample_dict.items():
    if args.name != '':
        if args.name not in name: continue

    # get sample list
    query = '"dataset={0}"'.format(loc)
    command = 'dasgoclient --query={0}'.format(query)
    print('...executing:\n', command)
    sample = subprocess.check_output(command, shell=True).decode().split('\n')[0]
    outfile.write("{0}_{1}:\n".format(name, args.year))

    # list all files in sample
    print('...sample:', sample)
    #print(" {0}".format(sample))
    
    query = '"file dataset={0}"'.format(sample)
    command = 'dasgoclient --query={0}'.format(query)
    try: 
        sample_files = subprocess.check_output(command, shell=True)
    except: 
        continue
    
    sample_files = sample_files.split('\n')[:-1]
    for sample_file in sample_files:
        if ((not sample_file) or (era not in sample_file)): continue
        command = 'edmFileUtil -d {0}'.format(sample_file)
        sample_file_loc = subprocess.check_output(command, shell=True)    
        print(sample_file_loc)
        outfile.write("  - " + sample_file_loc.strip('\n').replace('-site','') + "\n")

outfile.close()


