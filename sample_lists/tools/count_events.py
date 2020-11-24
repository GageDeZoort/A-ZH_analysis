import uproot
import yaml

with open("all_2018_samples.yaml", 'r') as stream:
    try:
        fileset = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

evt_total = 0
for dataset, datafiles in fileset.items():
    print("Dataset:", dataset)
    evt_sum = 0
    for datafile in datafiles:
        open_file = uproot.open(datafile)
        evt_sum += open_file['Events'].numentries
    print("   --> n_events =", evt_sum)
    evt_total += evt_sum

print("\n ==> Total Events:", evt_total)
