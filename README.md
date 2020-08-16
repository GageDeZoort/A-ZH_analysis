# Princeton A/ZH Framework
Framework for the A->ZH->lltt and ZH->lltt analyses. 
## Setup
This analysis framework is configured to work with the cmslpc-sl7 cluster, which you can ssh into with `ssh username>@cmslpc-sl7.fnal.gov`. For full details about setting up your computing environment, please see the USCMS [Software Setup Guide](https://uscms.org/uscms_at_work/computing/setup/setup_software.shtml). First, source the environment setup script:
```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh (for bash, .csh for cshell)
```
You'll need to set up the analysis in the work area of a CMSSW release:
```bash
cmsrel CMSSW_10_2_9
cd CMSSW_10_2_9/src
cmsenv
```
### pip
Each script has various dependencies, which you may or may not have installed in your Python3 environment. For example, you may need to install yaml like so:
```bash
pip install --user pyyaml
```

### conda
I find it helpful to keep my software organized in Conda environments. At Fermilab's recent [ML Hands-on Tutorial Session](https://github.com/FNALLPC/machine-learning-hats), they provide the following recipe for installing conda on the LPC:
```bash 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/nobackup/miniconda3.sh
bash $HOME/nobackup/miniconda3.sh -b -f -u -p $HOME/nobackup/miniconda3
source $HOME/nobackup/miniconda3/etc/profile.d/conda.sh
```
You can then create a conda virtual environment and install relevant packages within it, for example:
```bash
conda create --name AZH_env python=3.6
conda install -c conda-forge root # install root in conda env
conda install matplotlib          # install other packages via conda
conda install pip                 # add pip to the conda env

# pip install within your conda env:
/uscms/home/<username>/nobackup/miniconda3/envs/AZH/bin/pip install coffea
```
Note that you should *not* source cmsset_default.sh if you're working in a conda environment - minimally, this breaks root. 

## Job Submission
Jobs are submitted to the Condor cluster following examples listed in the [USCMS Batch Systems Guide](https://uscms.org/uscms_at_work/computing/setup/batch_systems.shtml) and the [Submitting Multiple Jobs Using HTCondor](http://chtc.cs.wisc.edu/multiple-jobs.shtml) page. 

## Pileup Re-weighting
[Utilities for Accessing Pileup Information for Data](https://twiki.cern.ch/twiki/bin/view/CMS/PileupJSONFileforData)
