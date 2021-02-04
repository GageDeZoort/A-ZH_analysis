import sys
from time import time

import yaml
import pandas as pd
import numpy as np
from coffea import processor

sys.path.append('../')
from processors.signal_processor import SignalProcessor

with open("sample_lists/samples_sync/AToZhToLLTauTau_M220_2018_samples.yaml", 'r') as stream:
    try:
        fileset = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

t0 = time()
out = processor.run_uproot_job (
    fileset,
    treename='Events',
    processor_instance=SignalProcessor(sync=True, categories='all',
                                       sample_list_dir='sample_lists'),
    executor=processor.futures_executor,
    executor_args={'workers':20, 'flatten': True, "nano": True},
    #chunksize=1000,
    #maxchunks=50,
)

print(out['cutflow_sync'].items())
print(out['cutflow'].items())

lumi = np.array(out['lumi'].value, dtype=int)
run = np.array(out['run'].value, dtype=int)
evt = np.array(out['evt'].value, dtype=int)

# extract SVfit quantities
np.savez(sys.argv[1],
         lumi=lumi, evt=evt,
         category=np.array(out['category'].value, dtype=int),
         pt1=np.array(out['l1_pt'].value, dtype=float),
         pt2=np.array(out['l2_pt'].value, dtype=float),
         pt3=np.array(out['t1_pt'].value, dtype=float),
         pt4=np.array(out['t2_pt'].value, dtype=float),
         eta1=np.array(out['l1_eta'].value, dtype=float),
         eta2=np.array(out['l2_eta'].value, dtype=float),
         eta3=np.array(out['t1_eta'].value, dtype=float),
         eta4=np.array(out['t2_eta'].value, dtype=float),
         phi1=np.array(out['l1_phi'].value, dtype=float),
         phi2=np.array(out['l2_phi'].value, dtype=float),
         phi3=np.array(out['t1_phi'].value, dtype=float),
         phi4=np.array(out['t2_phi'].value, dtype=float),
         mass3=np.array(out['t1_mass'].value, dtype=float),
         mass4=np.array(out['t2_mass'].value, dtype=float),
         METx=np.array(out['METx'].value, dtype=float),
         METy=np.array(out['METy'].value, dtype=float),
         METcov_00=np.array(out['METcov_00'].value, dtype=float),
         METcov_01=np.array(out['METcov_01'].value, dtype=float),
         METcov_10=np.array(out['METcov_00'].value, dtype=float),
         METcov_11=np.array(out['METcov_11'].value, dtype=float),
)

print("Processing took {0} s".format(time()-t0))

