import yaml
import uproot
import numpy as np

with open("../sample_lists/samples_sync/AToZhToLLTauTau_M220_2018_samples.yaml", 'r') as stream:
    try:
        fileset = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

for filename in fileset['AToZhToLLTauTau_M220_2018']:
    f = uproot.open(filename)
    print(filename)
    events = f['Events']
    evts = events.arrays()[b'event']
    lumis = events.arrays()[b'luminosityBlock']
    ntau = events.arrays()[b'nTau']
    check_evts = [248364, 248907, 249281, 249545, 249899, 250778]
    for evt in check_evts:
        ntau_evt = ntau[evts==evt]
        lumi = lumis[evts==evt]
        if (ntau_evt.shape[0] > 0):
            print(evt, lumi, ntau_evt)
