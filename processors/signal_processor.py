import os
import sys
import yaml
from time import time
import argparse

from functools import partial
import numpy as np
import mplhep as hep
import awkward1 as ak
import pandas as pd
import uproot

from coffea import hist, processor
from coffea.analysis_objects import JaggedCandidateArray
from coffea.nanoaod import NanoEvents
from coffea.lookup_tools import extractor
from uproot_methods import TLorentzVectorArray

class SignalProcessor(processor.ProcessorABC):
    def __init__(self, sync=False,  categories=[], 
                 checklist=pd.DataFrame([]),
                 sample_list_dir="../sample_lists"):

        # customize the 4l final states considered
        if categories == 'all':
            self.categories = {1: 'eeet', 2: 'eemt', 3: 'eett', 4: 'eeem',
                               5: 'mmet', 6: 'mmmt', 7: 'mmtt', 8: 'mmem'}
        else:
            self.categories = {i:cat for i, cat in enumerate(categories)}
        print("\n...running on", self.categories)
        
        # sync mode runs a subset of the full analysis
        self.sync = sync
        self.mode = 'sync' if self.sync else 'all'
        self.checklist = checklist # failing sync events to double-check

        # location of the samples, usually differentiates sync vs. all
        self.sample_list_dir = sample_list_dir
        
        # correct number of leptons in each final state
        self.correct_n_electrons = {'eeem': 3, 'eeet': 3, 'eemt': 2, 
                                    'eett': 2, 'mmem': 1, 'mmet': 1, 
                                    'mmmt': 0, 'mmtt': 0}
        self.correct_n_muons = {'eeem': 1, 'eeet': 0, 'eemt': 1, 
                                'eett': 0, 'mmem': 3, 'mmet': 2, 
                                'mmmt': 3, 'mmtt': 2}
        
        # histogram axes specify histo names, labels, and bin shapes
        category_axis = hist.Cat("category", "")
        dataset_axis = hist.Cat("dataset", "")
        
        pt1_axis = hist.Bin("pt1", "$p_T(l_1)$ [GeV]", 20, 0, 200)
        pt2_axis = hist.Bin("pt2", "$p_T(l_2)$ [GeV]", 20, 0, 200)
        pt3_axis = hist.Bin("pt3", "$p_T(t_1)$ [GeV]", 20, 0, 200)
        pt4_axis = hist.Bin("pt4", "$p_T(t_2)$ [GeV]", 20, 0, 200)
        
        eta1_axis = hist.Bin("eta1", "$\eta (l_1)$ [GeV]", 10, -5, 5)
        eta2_axis = hist.Bin("eta2", "$\eta (l_2)$ [GeV]", 10, -5, 5)
        eta3_axis = hist.Bin("eta3", "$\eta (t_1)$ [GeV]", 10, -5, 5)
        eta4_axis = hist.Bin("eta4", "$\eta (t_2)$ [GeV]", 10, -5, 5)
        
        phi1_axis = hist.Bin("phi1", "$\phi (l_1)$ [GeV]", 10, -np.pi, np.pi)
        phi2_axis = hist.Bin("phi2", "$\phi (l_2)$ [GeV]", 10, -np.pi, np.pi)
        phi3_axis = hist.Bin("phi3", "$\phi (t_1)$ [GeV]", 10, -np.pi, np.pi)
        phi4_axis = hist.Bin("phi4", "$\phi (t_2)$ [GeV]", 10, -np.pi, np.pi)
        
        mll_axis = hist.Bin("mll", "$m(l_1,l_2)$ [GeV]", 22, 40, 200)
        mtt_axis = hist.Bin("mtt", "$m(t_1,t_2)$ [GeV]", 40, 0, 400)
        m4l_axis = hist.Bin("m4l", "$m(l_1,l_2,t_1,t_2)$ [GeV]", 60, 0, 600)
    
        nbtag_axis = hist.Bin("nbtag", "$n_{btag}$", 5, 0, 5)
        njets_axis = hist.Bin("njets", "$n_{jets}$", 5, 0, 5)
        jpt1_axis = hist.Bin("jpt1", "$p_T(j_1)$ [GeV]", 20, 0, 200)
        jeta1_axis = hist.Bin("jeta1", "$\eta (j_1)$ [GeV]", 10, -5, 5)
        jphi1_axis = hist.Bin("jphi1", "$\phi (j_1)$ [GeV]", 10, -np.pi, np.pi)
        bpt1_axis = hist.Bin("bpt1", "$p_T(b_1)$ [GeV]", 20, 0, 200)
        beta1_axis = hist.Bin("beta1", "$\eta (b_1)$ [GeV]", 10, -5, 5)
        bphi1_axis = hist.Bin("bphi1", "$\phi (b_1)$ [GeV]", 10, -np.pi, np.pi)
        
        self._accumulator = processor.dict_accumulator({
            
            # event info, weights
            "sumw": processor.defaultdict_accumulator(float),
            "evt": processor.column_accumulator(np.array([])),
            "lumi": processor.column_accumulator(np.array([])),
            "run": processor.column_accumulator(np.array([])),
            
            # SVfit output
            "l1_pt": processor.column_accumulator(np.array([])),     
            "l2_pt": processor.column_accumulator(np.array([])),
            "t1_pt": processor.column_accumulator(np.array([])), 
            "t2_pt": processor.column_accumulator(np.array([])),
            "l1_eta": processor.column_accumulator(np.array([])),
            "l2_eta": processor.column_accumulator(np.array([])),
            "t1_eta": processor.column_accumulator(np.array([])),
            "t2_eta": processor.column_accumulator(np.array([])),
            "t1_mass": processor.column_accumulator(np.array([])),
            "t2_mass": processor.column_accumulator(np.array([])),
            "l1_phi": processor.column_accumulator(np.array([])),
            "l2_phi": processor.column_accumulator(np.array([])),
            "t1_phi": processor.column_accumulator(np.array([])),
            "t2_phi": processor.column_accumulator(np.array([])),
            "METx": processor.column_accumulator(np.array([])),  
            "METy": processor.column_accumulator(np.array([])),
            "METcov_00": processor.column_accumulator(np.array([])), 
            "METcov_01": processor.column_accumulator(np.array([])),
            "METcov_10": processor.column_accumulator(np.array([])),
            "METcov_11": processor.column_accumulator(np.array([])),
            "category": processor.column_accumulator(np.array([])),

            # histograms
            "pt1":   hist.Hist("Events", dataset_axis, category_axis, pt1_axis),
            "pt2":   hist.Hist("Events", dataset_axis, category_axis, pt2_axis),
            "pt3":   hist.Hist("Events", dataset_axis, category_axis, pt3_axis),
            "pt4":   hist.Hist("Events", dataset_axis, category_axis, pt4_axis),
            "eta1":  hist.Hist("Events", dataset_axis, category_axis, eta1_axis),
            "eta2":  hist.Hist("Events", dataset_axis, category_axis, eta2_axis),
            "eta3":  hist.Hist("Events", dataset_axis, category_axis, eta3_axis),
            "eta4":  hist.Hist("Events", dataset_axis, category_axis, eta4_axis),
            "phi1":  hist.Hist("Events", dataset_axis, category_axis, phi1_axis),
            "phi2":  hist.Hist("Events", dataset_axis, category_axis, phi2_axis),
            "phi3":  hist.Hist("Events", dataset_axis, category_axis, phi3_axis),
            "phi4":  hist.Hist("Events", dataset_axis, category_axis, phi4_axis),
            "mll":   hist.Hist("Events", dataset_axis, category_axis, mll_axis),
            "mtt":   hist.Hist("Events", dataset_axis, category_axis, mtt_axis),
            "m4l":   hist.Hist("Events", dataset_axis, category_axis, m4l_axis),
            "nbtag": hist.Hist("Events", dataset_axis, category_axis, nbtag_axis),
            "njets": hist.Hist("Events", dataset_axis, category_axis, njets_axis),
            
            # cutflow 
            'cutflow': processor.defaultdict_accumulator(
                partial(processor.defaultdict_accumulator, int)
            ),
            'cutflow_sync': processor.defaultdict_accumulator(
                partial(processor.defaultdict_accumulator, int)
            )
    })
        
    @property
    def accumulator(self): 
        return self._accumulator

    def check_events(self, evts):
        if not self.sync: return 0
        n_matches = 0
        for i, row in self.checklist.iterrows():
            n_matches += len(evts[(evts.run==row['run']) & 
                                  (evts.lumi==row['lumi']) &
                                  (evts.evt==row['evtid'])])
        return n_matches

    def fill_cutflow(self, label, N, N_sync=-1):
        self.output['cutflow'][self.dataset][label] += N
        if N_sync > -1:
            self.output['cutflow_sync'][self.dataset][label] += N_sync

    def trigger_selections(self, HLT, year, sync=False):

        if (sync):
            if (year in ['2017', '2018']): return HLT.Ele35_WPTight_Gsf
            if (year == '2016'): return HLT.Ele27_WPTight_Gsf

        if (year == '2016'):
            good_single = (HLT.IsoMu22 | HLT.IsoMu22_eta2p1 | HLT.IsoTkMu22 |
                           HLT.IsoTkMu22_eta2p1 | HLT.Ele25_eta2p1_WPTight_Gsf |
                           HLT.Ele27_eta2p1_WPTight_Gsf | HLT.IsoMu24 | 
                           HLT.IsoTkMu24 | HLT.IsoMu27)
            good_double = (HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ |
                           HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ |
                           HLT.Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ)
        elif (year == '2017') or (year == '2018'):
             good_single = (HLT.Ele27_WPTight_Gsf | HLT.Ele35_WPTight_Gsf |
                            HLT.Ele32_WPTight_Gsf | HLT_IsoMu24 | HLT_IsoMu27)
             
             good_double = (HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL | 
                            HLT.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ |
                            HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 |
                            HLT.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8)

        return (good_single | good_double)

    def print_taus(self, taus):
        for i, tau in enumerate(taus):
            print("tau{0}_pt: {1}".format(i, tau.pt))
            print("tau{0}_VSe: {1}".format(i, tau.rawDeepTau2017v2p1VSe))
            print("tau{0}_VSe: {1}".format(i, tau.idDeepTau2017v2p1VSe))

    def loose_tau_selections(self, taus, cutflow=False):

        self.fill_cutflow('initial taus', taus.shape[0],
                          N_sync=self.check_events(self.event_ids[taus.counts>0]))
        loose_taus = taus[(taus.pt > 20)]
        self.fill_cutflow('tau pt', taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[(np.abs(loose_taus.eta) < 2.3)]
        self.fill_cutflow('tau eta', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[(loose_taus.dz < 0.2)]
        self.fill_cutflow('tau dz', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[(loose_taus.idDecayModeNewDMs == 1)]
        self.fill_cutflow('tau decaymode', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[((loose_taus.decayMode != 5) & (loose_taus.decayMode != 6))]
        self.fill_cutflow('tau decaymodes', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        #(np.abs(taus.charge) == 1) &
        loose_taus = loose_taus[(loose_taus.idDeepTau2017v2p1VSe > 0)]
        self.fill_cutflow('tau vsjet', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[(loose_taus.idDeepTau2017v2p1VSmu > 0)] # Loose
        self.fill_cutflow('tau vsmu', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[(loose_taus.idDeepTau2017v2p1VSe > 3)]  # VLoose
        self.fill_cutflow('tau vse', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))

        if cutflow:
            enough_taus = (loose_taus.counts>0)
            self.fill_cutflow('loose taus', loose_taus[enough_taus].shape[0],
                              N_sync=self.check_events(self.event_ids[enough_taus]))

        return loose_taus

    def loose_muon_selections(self, muons, cutflow=False):
        loose_muons = muons[((muons.isTracker) | (muons.isGlobal)) & # DIFF
                            (muons.looseId | muons.mediumId | muons.tightId) &
                            (muons.dxy < 0.045) &
                            (muons.dz < 0.2) &
                            (muons.pt > 10) &
                            (np.abs(muons.eta) < 2.4)] # &
        #                   (muons.pfRelIso04_all < 0.25)]
        if cutflow: 
            enough_muons = (loose_muons.counts>0)
            self.fill_cutflow('loose muons', loose_muons[enough_muons].shape[0],
                              N_sync=self.check_events(self.event_ids[enough_muons]))
        return loose_muons
            
    def loose_electron_selections(self, electrons, cutflow=False):
        loose_electrons = electrons[(electrons.dxy < 0.045) &
                                    (electrons.dz  < 0.2) &
                                    (electrons.mvaFall17V2noIso_WP90 > 0.5) &
                                    (electrons.lostHits < 2) &
                                    (electrons.convVeto) &
                                    #(electrons.pfRelIso03_all < 0.2) &
                                    (electrons.pt > 10) &
                                    (np.abs(electrons.eta) < 2.5)]
        if cutflow: 
            enough_electrons = (loose_electrons.counts>0)
            self.fill_cutflow('loose electrons', loose_electrons[enough_electrons].shape[0],
                              N_sync=self.check_events(self.event_ids[enough_electrons]))
        return loose_electrons

    def loose_jet_selections(self, jets, year):
        
        if (year != '2017'):
            return jets[(jets.jetId >= 2) & # Tight
                        (jets.pt > 30) &
                        (np.abs(jets.eta) < 4.7) &
                        (((jets.pt < 50) & (jets.puId < 4)) | (jets.pt >= 50))]
            
        # remove noisy jets for 2017 samples
        else: return loose_jets[(loose_jets.pt > 20) &
                                (loose_jets.pt < 50) &
                                (np.abs(loose_jets.eta) > 2.65) &
                                (np.abs(loose_jets.eta) < 3.139)]

    def loose_bjet_selections(self, jets, year):
        bjet_thresholds = {'2016': 0.6321, '2017': 0.4941, '2018': 0.4184}
        bjet_flavor_threshold = 0.0614
        return jets[(jets.pt > 25) &
                    (np.abs(jets.eta) < 2.4) &
                    (jets.btagDeepB > bjet_thresholds[year]) &
                    (jets.btagDeepFlavB > bjet_flavor_threshold)]    
        
    def count_non_overlapped(self, leptons):
        ll = leptons.distincts()
        ll_dR = ll.i0.delta_r(ll.i1)
        ll_overlaps = (ll_dR < 0.3).sum()
        return leptons.counts - ll_overlaps        

    def dR_cut(self, lltt, cat, cutflow=False):
        dR_cuts = { 'ee': 0.3, 'em': 0.3, 'mm': 0.3, 'me': 0.3,
                    'et': 0.5, 'mt': 0.5, 'tt': 0.5 }
        dR_mask = ( (lltt.i0.delta_r(lltt.i1) > dR_cuts[cat[0]+cat[1]]) &
                    (lltt.i0.delta_r(lltt.i2) > dR_cuts[cat[0]+cat[2]]) &
                    (lltt.i0.delta_r(lltt.i3) > dR_cuts[cat[0]+cat[3]]) &
                    (lltt.i1.delta_r(lltt.i2) > dR_cuts[cat[1]+cat[2]]) &
                    (lltt.i1.delta_r(lltt.i3) > dR_cuts[cat[1]+cat[3]]) & 
                    (lltt.i2.delta_r(lltt.i3) > dR_cuts[cat[2]+cat[3]]) )
        lltt = lltt[dR_mask]
        if cutflow: self.fill_cutflow('dR overlap', lltt[lltt.counts>0].shape[0],
                                      N_sync=self.check_events(self.evt_ids[lltt.counts>0]))
        return lltt

    def n_lepton_veto(self, electron_counts, muon_counts, category):
        correct_n_electrons = {'eeem': 3, 'eeet': 3, 'eemt': 2, 'eett': 2,
                               'mmem': 1, 'mmet': 1, 'mmmt': 0, 'mmtt': 0}
        correct_n_muons = {'eeem': 1, 'eeet': 0, 'eemt': 1, 'eett': 0,
                           'mmem': 3, 'mmet': 2, 'mmmt': 3, 'mmtt': 2}
        return ( (electron_counts == correct_n_electrons[category]) & 
                 (muon_counts == correct_n_muons[category]) )

    def build_Z_cand(self, lltt, category, cutflow=False):
        
        # oppositely charged light leptons in the Z mass window
        lltt = lltt[ (lltt.i0.charge != lltt.i1.charge) &
                     (((lltt.i0 + lltt.i1).mass > 60) & 
                     ((lltt.i0 + lltt.i1).mass < 120))]
        
        mass_diffs = abs( (lltt.i0+lltt.i1).mass - 91.118 )
        min_mass_diffs = mass_diffs.min()
        closest_mass_mask = ((mass_diffs - min_mass_diffs) == 0)
        lltt = lltt[closest_mass_mask]
        
        if cutflow: self.fill_cutflow('Z cand', lltt[lltt.counts>0].shape[0],
                                      N_sync=self.check_events(self.evt_ids[lltt.counts>0]))
        return lltt
        
    def build_ditau_cand(self, lltt, category, cutflow=False):
        
        #lltt = lltt[(lltt.i2.charge * lltt.i3.charge == -1)]
        #if cutflow: self.fill_cutflow('ditau charge', lltt[lltt.counts>0].shape[0],
        #                              N_sync = self.check_events(self.evt_ids[lltt.counts>0]))

        if (category[2:] == 'mt'):
            lltt = lltt[(lltt.i3.idDeepTau2017v2p1VSmu > 7)]# & # Tight
                        #(lltt.i2.pfRelIso04_all < 0.15) &
                        #((lltt.i2.pt + lltt.i3.pt) > 40)] # &
                        #(lltt.i3.idAntiMu > 2) &
                        #(lltt.i3.idDeepTau2017v2p1VSjet > 15)] # Medium
                                
        elif (category[2:] == 'tt'):
            lltt = lltt[(lltt.i2.charge * lltt.i3.charge == -1) &
                        (lltt.i2.idDeepTau2017v2p1VSjet > 15) & # Medium
                        (lltt.i3.idDeepTau2017v2p1VSjet > 15) & # Medium
                        ((lltt.i2.pt + lltt.i3.pt) > 80)]
        
        elif (category[2:] == 'et'):
            lltt = lltt[(lltt.i2.mvaFall17V2noIso_WP80) &
                        (lltt.i2.pfRelIso03_all < 0.15) & 
                        #(lltt.i2.mvaFall17V2noIso_WP90) & # from ZH
                        (lltt.i3.idDeepTau2017v2p1VSe > 31) & # Tight
                        ((lltt.i2.pt + lltt.i3.pt) > 30)] 
        
        elif (category[2:] == 'em'):
            lltt = lltt[(lltt.i2.mvaFall17V2noIso_WP80) & 
                        #(lltt.i2.mvaFall17V2noIso_WP90) & # from ZH
                        (lltt.i2.pfRelIso03_all < 0.15) &
                        (lltt.i3.pfRelIso04_all < 0.15) &
                        ((lltt.i2.pt + lltt.i3.pt) > 20)]
                    
        lltt = lltt[(lltt.i2.pt + lltt.i3.pt).argmax()]
        if cutflow: self.fill_cutflow('ditau cand', lltt[lltt.counts>0].shape[0],
                                      N_sync = self.check_events(self.evt_ids[lltt.counts>0]))    
        return lltt

    def get_masses(self, lltt, cutflow=False):
        mll = (lltt.i0 + lltt.i1).mass
        mtt = (lltt.i2 + lltt.i3).mass
        m4l = (lltt.i0 + lltt.i1 + lltt.i2 + lltt.i3).mass
        if cutflow:
            self.fill_cutflow('final state', len(self.evt_ids[lltt.counts>0]),
                              N_sync = self.check_events(self.evt_ids[lltt.counts>0]))
        return mll, mtt, m4l

    ###############
    ## PROCESSOR ##
    ############### 
    def process(self, events):
        # grab dataset metadata
        self.dataset = events.metadata['dataset']
        year = self.dataset.split('_')[-1]
        self.output = self.accumulator.identity()
        print(year, self.dataset)

        # grab event id data
        self.event_ids = pd.DataFrame({'run': np.array(events.run, dtype=int),
                                       'lumi': np.array(events.luminosityBlock, dtype=int),
                                       'evt': np.array(events.event, dtype=int)})
        self.fill_cutflow('all events', len(events), 
                          N_sync = self.check_events(self.event_ids))

        # name data-taking eras, integrated lumis
        eras = {'2016': 'Summer16', '2017': 'Fall17', '2018': 'Autumn18'}
        lumi = {'2016': 35.9, '2017': 41.5, '2018': 59.7}

        # load properties of each sample
        with open("{0}/samples_{1}/{2}_properties.yaml"
                  .format(self.sample_list_dir,
                          self.mode, self.dataset), 'r') as stream:
            try:
                properties = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        xsec = float(properties[self.dataset]['xsec'])
        total_weight = float(properties[self.dataset]['total_weight'])
        sample_weight = lumi[year]*xsec/total_weight
        if ('data' in self.dataset): sample_weight = 1.
    
        #############
        ## FILTERS ##
        #############
        # calculate the MET filter
        flags = events.Flag
        MET_filter = (flags.goodVertices & flags.HBHENoiseFilter &
                      flags.HBHENoiseIsoFilter &
                      flags.EcalDeadCellTriggerPrimitiveFilter &
                      flags.BadPFMuonFilter & flags.ecalBadCalibFilter)

        # calculate the trigger_filter
        HLT = events.HLT
        trigger_filter = self.trigger_selections(HLT, year, sync=self.sync)
        
        # apply filters
        events = events[MET_filter & trigger_filter]
        self.event_ids = self.event_ids[MET_filter & trigger_filter]
        self.fill_cutflow('trigger filter', len(events),
                          N_sync = self.check_events(self.event_ids))

        ######################
        ## LOOSE SELECTIONS ##
        ######################
        # apply loose selections
        loose_taus = self.loose_tau_selections(events.Tau)
        loose_muons = self.loose_muon_selections(events.Muon)
        loose_electrons = self.loose_electron_selections(events.Electron)
        loose_jets = self.loose_jet_selections(events.Jet, year)
        loose_bjets = self.loose_bjet_selections(events.Jet, year)
        MET = events.MET

        # count electrons minus overlapped objects
        electron_counts = self.count_non_overlapped(loose_electrons)
        muon_counts = self.count_non_overlapped(loose_muons)

        ll_pairs = { 'ee': loose_electrons.distincts(), 
                     'mm': loose_muons.distincts() }
        tt_pairs = { 'mt': loose_muons.cross(loose_taus),
                     'et': loose_electrons.cross(loose_taus),
                     'em': loose_electrons.cross(loose_muons),
                     'tt': loose_taus.distincts() }        
        
        ###############################
        ## SELECTIONS (per category) ##
        ###############################
        for c, category in self.categories.items():

            # n_leptons veto 
            n_lepton_mask = self.n_lepton_veto(electron_counts, muon_counts, category)
            jets, bjets = loose_jets[n_lepton_mask], loose_bjets[n_lepton_mask]
            met = MET[n_lepton_mask]

            # track event ids on a per-category basis
            self.evt_ids = self.event_ids[n_lepton_mask]
            
            # form 4l final states
            ll = ll_pairs[category[:2]][n_lepton_mask]
            tt = tt_pairs[category[2:]][n_lepton_mask]
            lltt = ll.cross(tt)
            self.fill_cutflow('n_lepton veto', len(self.evt_ids),
                              N_sync = self.check_events(self.evt_ids))
            
            # build non-overlapped final state objects 
            lltt = self.dR_cut(ll.cross(tt), category, cutflow=True)
            lltt = self.build_Z_cand(lltt, category, cutflow=True)
            lltt = self.build_ditau_cand(lltt, category, cutflow=True)

            # apply b jet veto
            #self.evt_ids = self.evt_ids[bjets.counts==0]
            #lltt, met = lltt[bjets.counts==0], met[bjets.counts==0]
            #jets = jets[bjets.counts==0], 
            #bjets = bjets[bjets.counts==0]
            #self.fill_cutflow('bjet veto', lltt[lltt.counts>0].shape[0],
            #                  N_sync = self.check_events(self.evt_ids[lltt.counts>0]))

            # take only valid final states
            self.evt_ids = self.evt_ids[lltt.counts>0]
            lltt = lltt[lltt.counts>0]
            mll, mtt, m4l = self.get_masses(lltt, cutflow=True)
            
            
            #################
            ## FILL HISTOS ##
            #################
            self.output["evt"] += processor.column_accumulator(self.evt_ids['evt'].to_numpy()) 
            self.output["lumi"] += processor.column_accumulator(self.evt_ids['lumi'].to_numpy())
            self.output["run"] += processor.column_accumulator(self.evt_ids['run'].to_numpy())
            
            pt1, pt2 = lltt.i0.pt.flatten(), lltt.i1.pt.flatten()
            pt3, pt4 = lltt.i2.pt.flatten(), lltt.i3.pt.flatten()
            self.output["l1_pt"] += processor.column_accumulator(pt1)
            self.output["l2_pt"] += processor.column_accumulator(pt2)
            self.output["t1_pt"] += processor.column_accumulator(pt3)
            self.output["t2_pt"] += processor.column_accumulator(pt4)

            eta1, eta2 = lltt.i0.eta.flatten(), lltt.i1.eta.flatten()
            eta3, eta4 = lltt.i2.eta.flatten(), lltt.i3.eta.flatten()
            self.output["l1_eta"] += processor.column_accumulator(ak.to_numpy(eta1))
            self.output["l2_eta"] += processor.column_accumulator(ak.to_numpy(eta2))
            self.output["t1_eta"] += processor.column_accumulator(ak.to_numpy(eta3))
            self.output["t2_eta"] += processor.column_accumulator(ak.to_numpy(eta4))
            
            phi1, phi2 = lltt.i0.phi.flatten(), lltt.i1.phi.flatten()
            phi3, phi4 = lltt.i2.phi.flatten(), lltt.i3.phi.flatten()
            self.output["l1_phi"] += processor.column_accumulator(ak.to_numpy(phi1))
            self.output["l2_phi"] += processor.column_accumulator(ak.to_numpy(phi2))
            self.output["t1_phi"] += processor.column_accumulator(ak.to_numpy(phi3))
            self.output["t2_phi"] += processor.column_accumulator(ak.to_numpy(phi4))

            mass3, mass4 = lltt.i2.mass.flatten(), lltt.i3.mass.flatten()
            self.output["t1_mass"] += processor.column_accumulator(ak.to_numpy(mass3))
            self.output["t2_mass"] += processor.column_accumulator(ak.to_numpy(mass4))
            
            self.output["METx"] += processor.column_accumulator(ak.to_numpy((met.pt*np.cos(met.phi)).flatten()))
            self.output["METy"] += processor.column_accumulator(ak.to_numpy((met.pt*np.sin(met.phi)).flatten()))
            self.output["METcov_00"] += processor.column_accumulator(ak.to_numpy(met.covXX.flatten()))
            self.output["METcov_01"] += processor.column_accumulator(ak.to_numpy(met.covXY.flatten()))
            self.output["METcov_10"] += processor.column_accumulator(ak.to_numpy(met.covXY.flatten()))
            self.output["METcov_11"] += processor.column_accumulator(ak.to_numpy(met.covYY.flatten()))
            self.output["category"] += processor.column_accumulator(c * np.ones(len(pt1)) )

            self.output["pt1"].fill(dataset=self.dataset, category=category, pt1=pt1, weight=sample_weight*np.ones(len(pt1)))
            self.output["pt2"].fill(dataset=self.dataset, category=category, pt2=pt2, weight=sample_weight*np.ones(len(pt2)))
            self.output["pt3"].fill(dataset=self.dataset, category=category, pt3=pt3, weight=sample_weight*np.ones(len(pt3)))
            self.output["pt4"].fill(dataset=self.dataset, category=category, pt4=pt4, weight=sample_weight*np.ones(len(pt4)))
            
            self.output["eta1"].fill(dataset=self.dataset, category=category, eta1=eta1, weight=sample_weight*np.ones(len(eta1)))
            self.output["eta2"].fill(dataset=self.dataset, category=category, eta2=eta2, weight=sample_weight*np.ones(len(eta2)))
            self.output["eta3"].fill(dataset=self.dataset, category=category, eta3=eta3, weight=sample_weight*np.ones(len(eta3)))
            self.output["eta4"].fill(dataset=self.dataset, category=category, eta4=eta4, weight=sample_weight*np.ones(len(eta4)))
             
            self.output["phi1"].fill(dataset=self.dataset, category=category, phi1=phi1, weight=sample_weight*np.ones(len(phi1)))
            self.output["phi2"].fill(dataset=self.dataset, category=category, phi2=phi2, weight=sample_weight*np.ones(len(phi2)))
            self.output["phi3"].fill(dataset=self.dataset, category=category, phi3=phi3, weight=sample_weight*np.ones(len(phi3)))
            self.output["phi4"].fill(dataset=self.dataset, category=category, phi4=phi4, weight=sample_weight*np.ones(len(phi4)))
            
            self.output["mll"].fill(dataset=self.dataset, category=category, mll=mll.flatten(), weight=sample_weight*np.ones(len(mll.flatten())))
            self.output["mtt"].fill(dataset=self.dataset, category=category, mtt=mtt.flatten(), weight=sample_weight*np.ones(len(mtt.flatten())))
            self.output["m4l"].fill(dataset=self.dataset, category=category, m4l=m4l.flatten(), weight=sample_weight*np.ones(len(m4l.flatten())))
        
            #nbtag = good_bjets.counts.flatten()
            #output["nbtag"].fill(dataset=dataset, category=category, nbtag=nbtag, weight=sample_weight*np.ones(len(nbtag)))
            #njets = good_jets.counts.flatten()
            #output["njets"].fill(dataset=dataset, category=category, njets=njets, weight=sample_weight*np.ones(len(njets)))

        return self.output

    def postprocess(self, accumulator):
        return accumulator
