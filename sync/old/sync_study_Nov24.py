import sys
from time import time

import yaml
import pandas as pd
import numpy as np
from coffea import processor

from processors.signal_processor import SignalProcessor

with open("sample_lists/samples_sync/AToZhToLLTauTau_M220_2016_samples.yaml", 'r') as stream:
    try:
        fileset = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

checkdata = {'run': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             'lumi': [1, 3, 1790, 1791, 1791, 1791, 1791, 1791, 1791, 1791, 1791],
             'evt': [121, 518, 328059, 328222, 328235, 328242, 328253, 328270, 328270, 328270, 328386]}
checklist = pd.DataFrame(checkdata)

t0 = time()
out = processor.run_uproot_job(
    fileset,
    treename='Events',
    processor_instance=SignalProcessor(mode='sync', categories=['eemt'],
                                       checklist=checklist),
    executor=processor.futures_executor,
    executor_args={'workers':20, 'flatten': True, "nano": True},
    #chunksize=1000,
    #maxchunks=50,
)

lumi = np.array(out['lumi'].value, dtype=int)
run = np.array(out['run'].value, dtype=int)
evt = np.array(out['evt'].value, dtype=int)
print(run, lumi, evt)

if (len(sys.argv)==1):
    filename = 'sync_out.csv'
else:
    filename = sys.argv[1]

sync_file = open(filename, 'w')
sync_file.write('run,lumi,evtid\n')
for i, e in enumerate(evt):
   sync_file.write('{0:d},{1:d},{2:d}\n'
                   .format(run[i], lumi[i], evt[i]))
sync_file.close()

print("Processing took {0} s".format(time()-t0))

