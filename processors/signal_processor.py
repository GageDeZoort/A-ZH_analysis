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
import ROOT

from coffea import hist, processor
from coffea.analysis_objects import JaggedCandidateArray
from coffea.nanoaod import NanoEvents
from coffea.lookup_tools import extractor
from uproot_methods import TLorentzVectorArray

class SignalProcessor(processor.ProcessorABC):
    def __init__(self, sync=False,  categories=[], 
                 checklist=pd.DataFrame([]),
                 sample_list_dir="../sample_lists"):

        # load in fastmtt
        fastmtt_dir = '../svfit/fastmtt/'
        for basename in ['MeasuredTauLepton', 'svFitAuxFunctions', 'FastMTT']:
            path = fastmtt_dir + basename
            if os.path.isfile("{0:s}_cc.so".format(path)):
                ROOT.gInterpreter.ProcessLine(".L {0:s}_cc.so".format(path))
            else:
                ROOT.gInterpreter.ProcessLine(".L {0:s}.cc++".format(path))
        
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
        self.princeton_exclusive = np.array([248633, 250132, 250374, 256311, 2568862, 259595, 395373, 488027, 490292, 491592])
        
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
        particle_axis = hist.Cat("particle", "")
        
        pt_axis = hist.Bin("pt", "$p_T$ [GeV]", 20, 0, 200)        
        eta_axis = hist.Bin("eta", "$\eta$ [GeV]", 10, -5, 5)        
        phi_axis = hist.Bin("phi", "$\phi$ [GeV]", 10, -np.pi, np.pi)
        
        mll_axis = hist.Bin("mll", "$m(l_1,l_2)$ [GeV]", 40, 0, 200)
        mtt_axis = hist.Bin("mtt", "$m(t_1,t_2)$ [GeV]", 30, 0, 300)
        mA_axis  = hist.Bin("mA", "$m_A$ [GeV]", 40, 0, 400)
        mass_type_axis = hist.Cat("mass_type", "")

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
            "cat": processor.column_accumulator(np.array([])),
            "mll_array": processor.column_accumulator(np.array([])),
            "msv_cons_array": processor.column_accumulator(np.array([])),
            "m_mumu": processor.column_accumulator(np.array([])), 

            # histograms
            "pt": hist.Hist("Events", dataset_axis, category_axis, pt_axis, particle_axis),
            "eta": hist.Hist("Events", dataset_axis, category_axis, eta_axis, particle_axis),
            "phi": hist.Hist("Events", dataset_axis, category_axis, phi_axis, particle_axis),
            "mll": hist.Hist("Events", dataset_axis, category_axis, mll_axis),
            "mtt": hist.Hist("Events", dataset_axis, category_axis, mtt_axis, mass_type_axis),
            "m4l": hist.Hist("Events", dataset_axis, category_axis, mA_axis,  mass_type_axis),
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

    def check_events(self, evts, output=True):
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

    def trigger_path(self, HLT, year, category, sync=False):

        if (sync):
            if (year in ['2017', '2018']): 
                if (category[:2]=='ee'): 
                    return HLT.Ele35_WPTight_Gsf
                elif (category[:2]=='mm'): 
                    return HLT.IsoMu27
            elif (year == '2016'): return HLT.Ele27_WPTight_Gsf

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

    def loose_tau_selections(self, taus, cutflow=False):

        self.fill_cutflow('initial taus', taus.shape[0],
                          N_sync=self.check_events(self.event_ids[taus.counts>0]))
        loose_taus = taus[(taus.pt > 20)]
        self.fill_cutflow('tau pt', taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[(np.abs(loose_taus.eta) < 2.3)]
        self.fill_cutflow('tau eta', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[(np.abs(loose_taus.dz) < 0.2)]
        self.fill_cutflow('tau dz', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[(loose_taus.idDecayModeNewDMs == 1)]
        self.fill_cutflow('tau decaymode', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[((loose_taus.decayMode != 5) & (loose_taus.decayMode != 6))]
        self.fill_cutflow('tau decaymodes', loose_taus.shape[0],
                          N_sync=self.check_events(self.event_ids[loose_taus.counts>0]))
        loose_taus = loose_taus[(loose_taus.idDeepTau2017v2p1VSjet > 0)]
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
                            (np.abs(muons.dxy) < 0.045) &
                            (np.abs(muons.dz) < 0.2) &
                            (muons.pt > 10) &
                            (np.abs(muons.eta) < 2.4)]  #&
                            #(muons.pfRelIso04_all < 0.25)]
        if cutflow: 
            enough_muons = (loose_muons.counts>0)
            self.fill_cutflow('loose muons', loose_muons[enough_muons].shape[0],
                              N_sync=self.check_events(self.event_ids[enough_muons]))
        return loose_muons
            
    def loose_electron_selections(self, electrons, cutflow=False):
        loose_electrons = electrons[(np.abs(electrons.dxy) < 0.045) &
                                    (np.abs(electrons.dz)  < 0.2) &
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
        return leptons.counts #- ll_overlaps        

    def dR_cut(self, lltt, cat, cutflow=False):
        dR_cuts = { 'ee': 0.3, 'em': 0.3, 'mm': 0.3, 'me': 0.3,
                    'et': 0.5, 'mt': 0.5, 'tt': 0.5 }
        dR_mask = ( (lltt.i0.delta_r(lltt.i1) > dR_cuts[cat[0]+cat[1]]) &
                    (lltt.i0.delta_r(lltt.i2) > dR_cuts[cat[0]+cat[2]]) &
                    (lltt.i0.delta_r(lltt.i3) > dR_cuts[cat[0]+cat[3]]) &
                    (lltt.i1.delta_r(lltt.i2) > dR_cuts[cat[1]+cat[2]]) &
                    (lltt.i1.delta_r(lltt.i3) > dR_cuts[cat[1]+cat[3]]) & 
                    (lltt.i2.delta_r(lltt.i3) > dR_cuts[cat[2]+cat[3]]) )
        #print('cat[0]={},cat[1]={}'.format(cat[0],cat[1]),
        #      '\n - dr: ', lltt.i0.delta_r(lltt.i1), 
        #      '\n - dr_cut: ', dR_cuts[cat[0]+cat[1]],
        #      '\n ==> ', (lltt.i0.delta_r(lltt.i1) > dR_cuts[cat[0]+cat[2]]))
        #print('cat[2]={}, cat[3]={}'.format(cat[2], cat[3]),
        #      '\n - dr: ', lltt.i2.delta_r(lltt.i3),
        #      '\n - dr_cut: ', dR_cuts[cat[2]+cat[3]],
        #      '\n ==> ', (lltt.i2.delta_r(lltt.i3) > dR_cuts[cat[2]+cat[3]]) )
        
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

    def trigger_filter(self, lltt, trig_obj, category, cutflow=False):
 
        lltt_trig = lltt.cross(trig_obj)
        l1_dR_matched = (lltt_trig.i0.delta_r(lltt_trig.i4) < 0.5)
        l2_dR_matched = (lltt_trig.i1.delta_r(lltt_trig.i4) < 0.5)
        filter_bit = ((lltt_trig.i4.filterBits & 2) > 0)

        if (category[:2] == 'ee'): 
            pt_min, eta_max = 36, 2.1
        if (category[:2] == 'mm'): 
            pt_min, eta_max = 28, 2.1
            filter_bit = ((filter_bit | ((lltt_trig.i4.filterBits & 8)) > 0)) 
            
        l1_matches = lltt_trig[((l1_dR_matched) & 
                                (lltt_trig.i0.pt > pt_min) &
                                filter_bit).astype(bool)]
        l2_matches = lltt_trig[((l2_dR_matched) & 
                                (lltt_trig.i1.pt > pt_min) & 
                                filter_bit).astype(bool)]        

        trigger_match = ((l1_matches.counts > 0) | (l2_matches.counts > 0))
        trigger_match_1 = l1_matches.counts > 0
        trigger_match_2 = l2_matches.counts > 0

        lltt = lltt[trigger_match]
        self.evt_ids = self.evt_ids[trigger_match]
        self.met = self.met[trigger_match]

        if cutflow: self.fill_cutflow('trigger filter', lltt[lltt.counts>0].shape[0],
                                      N_sync=self.check_events(self.evt_ids[lltt.counts>0]))
        return lltt

    def build_ditau_cand(self, lltt, category, cutflow=False):
        
        #lltt = lltt[(lltt.i2.charge * lltt.i3.charge == -1)]
        #if cutflow: self.fill_cutflow('ditau charge', lltt[lltt.counts>0].shape[0],
        #                              N_sync = self.check_events(self.evt_ids[lltt.counts>0]))

        if (category[2:] == 'mt'):
            lltt = lltt[(lltt.i3.idDeepTau2017v2p1VSmu > 7)] #  & # Tight
                        #(lltt.i2.pfRelIso04_all < 0.15) &
                        #((lltt.i2.pt + lltt.i3.pt) > 40)] # &
                        #(lltt.i3.idAntiMu > 2) &
                        #(lltt.i3.idDeepTau2017v2p1VSjet > 15)] # Medium
                                
        elif (category[2:] == 'tt'):
            lltt = lltt#(lltt.i2.charge * lltt.i3.charge == -1) &
                        #(lltt.i2.idDeepTau2017v2p1VSjet > 15) & # Medium
                        #(lltt.i3.idDeepTau2017v2p1VSjet > 15) & # Medium
                        #((lltt.i2.pt + lltt.i3.pt) > 80)]
        
        elif (category[2:] == 'et'):
            lltt = lltt[#(lltt.i2.mvaFall17V2noIso_WP80) &
                        #(lltt.i2.pfRelIso03_all < 0.15) & 
                        #(lltt.i2.mvaFall17V2noIso_WP90) & # from ZH
                        (lltt.i3.idDeepTau2017v2p1VSe > 31)] #& # Tight
                        #((lltt.i2.pt + lltt.i3.pt) > 30)] 
        
        elif (category[2:] == 'em'):
            lltt = lltt#[#(lltt.i2.mvaFall17V2noIso_WP80)]# & 
                        #(lltt.i2.pfRelIso03_all < 0.15) &
                        #(lltt.i3.pfRelIso04_all < 0.15) &
                        #((lltt.i2.pt + lltt.i3.pt) > 20)]
                    
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

    def run_fastmtt(self, lltt, met, category, cutflow=False):        
        
        # choose the correct lepton mass
        ele_mass, mu_mass = 0.511*10**-3, 0.105
        l_mass = ele_mass if category[:2] == 'ee' else mu_mass
        
        # flatten the final state leptons, assign to 4-vector arrays
        l1_p4_array = TLorentzVectorArray.from_ptetaphim(lltt.i0.pt.flatten(), 
                                                         lltt.i0.eta.flatten(), 
                                                         lltt.i0.phi.flatten(), 
                                                         l_mass*np.ones(len(lltt)))
        l2_p4_array = TLorentzVectorArray.from_ptetaphim(lltt.i1.pt.flatten(), 
                                                         lltt.i1.eta.flatten(), 
                                                         lltt.i1.phi.flatten(), 
                                                         l_mass*np.ones(len(lltt)))
        
        # choose the correct tau decay modes
        e_decay = ROOT.MeasuredTauLepton.kTauToElecDecay
        m_decay  = ROOT.MeasuredTauLepton.kTauToMuDecay
        had_decay = ROOT.MeasuredTauLepton.kTauToHadDecay
        if (category[2:]=='et'): t1_decay, t2_decay = e_decay, had_decay
        if (category[2:]=='em'): t1_decay, t2_decay = e_decay, m_decay
        if (category[2:]=='mt'): t1_decay, t2_decay = m_decay, had_decay
        if (category[2:]=='tt'): t1_decay, t2_decay = had_decay, had_decay

        # flatten the final state taus, assign to 4-vector arrays
        t1_p4_array = TLorentzVectorArray.from_ptetaphim(lltt.i2.pt.flatten(), 
                                                         lltt.i2.eta.flatten(),
                                                         lltt.i2.phi.flatten(), 
                                                         lltt.i2.mass.flatten())
        t2_p4_array = TLorentzVectorArray.from_ptetaphim(lltt.i3.pt.flatten(), 
                                                         lltt.i3.eta.flatten(),
                                                         lltt.i3.phi.flatten(), 
                                                         lltt.i3.mass.flatten())
        
        # flatten MET arrays 
        metx = met.pt*np.cos(met.phi).flatten() 
        mety = met.pt*np.sin(met.phi).flatten()
        metcov00, metcov11 = met.covXX.flatten(), met.covYY.flatten()
        metcov01, metcov10 = met.covXY.flatten(), met.covXY.flatten()

        # loop to calculate A mass
        N = len(t1_p4_array)
        tt_corr_masses, tt_cons_masses = np.zeros(N), np.zeros(N)
        A_corr_masses, A_cons_masses = np.zeros(N), np.zeros(N)

        for i in range(N):

            metcov = ROOT.TMatrixD(2,2)        
            metcov[0][0], metcov[1][1] = metcov00[i], metcov11[i]
            metcov[0][1], metcov[1][0] = metcov01[i], metcov10[i]
        
            tau_vector = ROOT.std.vector('MeasuredTauLepton')
            tau_pair = tau_vector()
            t1 = ROOT.MeasuredTauLepton(t1_decay, 
                                        t1_p4_array[i].pt,
                                        t1_p4_array[i].eta,
                                        t1_p4_array[i].phi,
                                        t1_p4_array[i].mass)
            t2 = ROOT.MeasuredTauLepton(t2_decay, 
                                        t2_p4_array[i].pt,
                                        t2_p4_array[i].eta,
                                        t2_p4_array[i].phi,
                                        t2_p4_array[i].mass)
            tau_pair.push_back(t1)
            tau_pair.push_back(t2)

            # run SVfit algorithm
            fastmtt = ROOT.FastMTT()
            fastmtt.run(tau_pair, metx[i], mety[i], metcov, False) # unconstrained
            tt_corr = fastmtt.getBestP4()
            tt_corr_p4 = ROOT.TLorentzVector()
            tt_corr_p4.SetPtEtaPhiM(tt_corr.Pt(), tt_corr.Eta(),
                                    tt_corr.Phi(), tt_corr.M())
            
            fastmtt.run(tau_pair, metx[i], mety[i], metcov, True) # constrained
            tt_cons = fastmtt.getBestP4()
            tt_cons_p4 = ROOT.TLorentzVector()
            tt_cons_p4.SetPtEtaPhiM(tt_cons.Pt(), tt_cons.Eta(),
                                    tt_cons.Phi(), tt_cons.M())

            tt_corr_masses[i] = tt_corr_p4.M()
            tt_cons_masses[i] = tt_cons_p4.M()

            l1, l2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
            l1.SetPtEtaPhiM(l1_p4_array[i].pt, l1_p4_array[i].eta,
                            l1_p4_array[i].phi, l1_p4_array[i].mass)
            l2.SetPtEtaPhiM(l2_p4_array[i].pt, l2_p4_array[i].eta,
                            l2_p4_array[i].phi, l2_p4_array[i].mass)
            A_corr_p4 = (l1 + l2 + tt_corr_p4)
            A_cons_p4 = (l1 + l2 + tt_cons_p4)
            A_corr_masses[i] = A_corr_p4.M()
            A_cons_masses[i] = A_cons_p4.M()
            
        return tt_corr_masses, tt_cons_masses, A_corr_masses, A_cons_masses

    def check_princeton_exclusive_loose(self, loose_electrons, loose_muons, loose_taus):
        for i in range(len(self.princeton_exclusive)):
            #test_run = self.princeton_exclusive[i,1]
            #test_lumi = self.princeton_exclusive[i,2]
            test_evt = self.princeton_exclusive[i]
            test_mask = (self.event_ids['evt'].to_numpy() == test_evt)# &
                         #(self.event_ids['lumi'].to_numpy() == test_lumi) &
                         #(self.event_ids['run'].to_numpy() == test_run))
            
            e = loose_electrons[test_mask]
            m = loose_muons[test_mask]
            t = loose_taus[test_mask]
            if (len(e) > 0):
                #print("----- (run={}, lumi={}, evt={}) -----"
                #      .format(test_run, test_lumi, test_evt))
                print("----- (evt={}) -----"
                      .format(test_evt))
                print("--> e: pt={}".format(e.pt))
                print("       eta={}".format(e.eta))
                print("       phi={}".format(e.phi))
                print("       mass={}".format(e.mass))
                print("       mvaFall17V2noIso_WP90={}".format(e.mvaFall17V2noIso_WP90))
                print("       lostHits={}, convVeto={}".format(e.lostHits, e.convVeto))
                print("--> m: pt={}".format(m.pt))
                print("       eta={}".format(m.eta))
                print("       phi={}".format(m.phi))
                print("       mass={}".format(m.mass))
                print("       isTracker={}, isGlobal={}"
                      .format(m.isTracker, m.isGlobal))
                print("       looseId={}, mediumId={}, tightId={}"
                      .format(m.looseId, m.mediumId, m.tightId))
                print("--> t: pt={}".format(t.pt))
                print("       eta={}".format(t.eta))
                print("       phi={}".format(t.phi))
                print("       mass={}".format(t.mass))
                print("       rawDeepTau2017v2p1VSe={}".format(t.rawDeepTau2017v2p1VSe))
                print("       rawDeepTau2017v2p1VSmu={}".format(t.rawDeepTau2017v2p1VSmu))
                print("       rawDeepTau2017v2p1VSjet={}".format(t.rawDeepTau2017v2p1VSjet))
                print("       decayMode={}, idDecayModeNewDMs={}"
                      .format(t.decayMode, t.idDecayModeNewDMs))
                
    def check_princeton_exclusive_fs(self, lltt):
        for i in range(len(self.princeton_exclusive)):
            #test_run = self.princeton_exclusive[i,1]
            #test_lumi = self.princeton_exclusive[i,2]
            test_evt = self.princeton_exclusive[i]
            test_mask = (self.evt_ids['evt'].to_numpy() == test_evt) #&
                         #(self.evt_ids['lumi'].to_numpy() == test_lumi) &
                         #(self.evt_ids['run'].to_numpy() == test_run))
            
            test_4l = lltt[test_mask]
            if (len(test_4l) > 0):
                #print(" SELECTED: (run={}, lumi={}, evt={})"
                #      .format(test_run, test_lumi, test_evt))
                print("SELECTED: evt={}"
                      .format(test_evt))
                print("--> e1: p4=({}, {}, {}, {})"
                      .format(test_4l.i0.pt, test_4l.i0.eta,
                              test_4l.i0.phi, test_4l.i0.mass))
                print("        mvaFall17V2noIso_WP90={}"
                      .format(test_4l.i0.mvaFall17V2noIso_WP90))
                print("        lostHits={}, convVeto={}"
                          .format(test_4l.i0.lostHits, test_4l.i0.convVeto))
                print("--> e2: p4=({}, {}, {}, {})"
                      .format(test_4l.i1.pt, test_4l.i1.eta,
                              test_4l.i1.phi, test_4l.i1.mass))
                print("        mvaFall17V2noIso_WP90={}"
                      .format(test_4l.i1.mvaFall17V2noIso_WP90))
                print("        lostHits={}, _convVeto={}"
                      .format(test_4l.i1.lostHits, test_4l.i1.convVeto))
                print("--> tau(mu): p4=({}, {}, {}, {})"
                      .format(test_4l.i2.pt, test_4l.i2.eta,
                              test_4l.i2.phi, test_4l.i2.mass))
                print("             isTracker={}, isGlobal={}"
                      .format(test_4l.i2.isTracker, test_4l.i2.isGlobal))
                print("             looseId={}, mediumId={}, tightId={}"
                      .format(test_4l.i2.looseId, test_4l.i2.mediumId, test_4l.i2.tightId))
                print("--> tau(had): p4=({}, {}, {}, {})"
                      .format(test_4l.i3.pt, test_4l.i3.eta,
                              test_4l.i3.phi, test_4l.i3.mass))
                print("              rawDeepTau2017v2p1VSe={}"
                      .format(test_4l.i3.rawDeepTau2017v2p1VSe))
                print("              rawDeepTau2017v2p1VSmu={}"
                      .format(test_4l.i3.rawDeepTau2017v2p1VSmu))
                print("              rawDeepTau2017v2p1VSjet={}"
                      .format(test_4l.i3.rawDeepTau2017v2p1VSjet))
                print("              decayMode={}, idDecayModeNewDMs={}"
                      .format(test_4l.i3.decayMode, test_4l.i3.idDecayModeNewDMs))
                

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
        
        # calculate PV quality filter
        pv = events.PV
        pv_filter = ((pv.ndof > 4) &
                     (abs(pv.z) < 24) & 
                     (np.sqrt(pv.x**2 + pv.y**2) < 2))
        
        # apply filters
        events = events[MET_filter & pv_filter]
        self.event_ids = self.event_ids[MET_filter & 
                                        pv_filter]
        self.fill_cutflow('MET, pv filters', len(events),
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
        trigger_objects = events.TrigObj
        
        #if category=='eemt':
        #    self.check_princeton_exclusive_loose(loose_electrons, loose_muons, loose_taus)

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
            # identify correct trigger path
            HLT = events.HLT
            trigger_path = self.trigger_path(HLT, year, category, sync=self.sync)

            # n_leptons veto 
            n_lepton_mask = self.n_lepton_veto(electron_counts, muon_counts, category)
            n_lepton_mask = n_lepton_mask & trigger_path # combine with trigger path
            jets, bjets = loose_jets[n_lepton_mask], loose_bjets[n_lepton_mask]
            self.met = MET[n_lepton_mask]
            trig_obj = trigger_objects[n_lepton_mask]

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
            lltt = self.trigger_filter(lltt, trig_obj, category, cutflow=True)
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
            self.met = self.met[lltt.counts>0]
            lltt = lltt[lltt.counts>0]
            mll, mtt, m4l = self.get_masses(lltt, cutflow=True)
            msv, msv_cons, mA_corr, mA_cons = self.run_fastmtt(lltt, self.met, category)

            #################
            ## FILL HISTOS ##
            #################
            self.output["evt"] += processor.column_accumulator(self.evt_ids['evt'].to_numpy()) 
            self.output["lumi"] += processor.column_accumulator(self.evt_ids['lumi'].to_numpy())
            self.output["run"] += processor.column_accumulator(self.evt_ids['run'].to_numpy())
            self.output["cat"] += processor.column_accumulator(np.array([category 
                                                                         for _ in range(len(self.evt_ids))]))

            if category=='eemt':
                self.check_princeton_exclusive_fs(lltt)
            
            pts = [lltt.i0.pt.flatten(), lltt.i1.pt.flatten(),
                   lltt.i2.pt.flatten(), lltt.i3.pt.flatten()]
            etas = [lltt.i0.eta.flatten(), lltt.i1.eta.flatten(),
                    lltt.i2.eta.flatten(), lltt.i3.eta.flatten()]
            phis = [lltt.i0.phi.flatten(), lltt.i1.phi.flatten(),
                    lltt.i2.phi.flatten(), lltt.i3.phi.flatten()]
            particle_nums = ["$l_1$", "$l_2$", "$t_1$", "$t_2$"]
            for i, pnum in enumerate(particle_nums):
                
                self.output["pt"].fill(dataset=self.dataset, category=category, pt=pts[i], 
                                       particle=pnum, weight=sample_weight*np.ones(len(pts[i])))
                self.output["eta"].fill(dataset=self.dataset, category=category, eta=etas[i], 
                                        particle=pnum, weight=sample_weight*np.ones(len(etas[i])))
                self.output["phi"].fill(dataset=self.dataset, category=category, phi=phis[i],
                                        particle=pnum, weight=sample_weight*np.ones(len(phis[i])))
            
            if (category[:2]=='mm'): self.output["m_mumu"] += processor.column_accumulator(np.array(mll.flatten()))
            self.output["mll_array"] += processor.column_accumulator(np.array(mll.flatten()))
            self.output["msv_cons_array"] += processor.column_accumulator(np.array(msv_cons.flatten()))
            
            self.output["mll"].fill(dataset=self.dataset, category=category, mll=mll.flatten(), 
                                    weight=sample_weight*np.ones(len(mll.flatten())))
            self.output["mtt"].fill(dataset=self.dataset, category=category, mass_type="$m_{tt}$", 
                                    mtt=mtt.flatten(), weight=sample_weight*np.ones(len(mtt.flatten())))
            self.output["mtt"].fill(dataset=self.dataset, category=category, mass_type="$m_{fastmtt}$", 
                                    mtt=msv.flatten(), weight=sample_weight*np.ones(len(msv.flatten())))
            self.output["m4l"].fill(dataset=self.dataset, category=category, mass_type='$m_{4l}$', 
                                    mA=m4l.flatten())
            self.output["m4l"].fill(dataset=self.dataset, category=category, mass_type='$m_A^{corr}$', 
                                    mA=mA_corr.flatten())
            self.output["m4l"].fill(dataset=self.dataset, category=category, mass_type='$m_A^{cons}$', 
                                    mA=mA_cons.flatten())

            #nbtag = good_bjets.counts.flatten()
            #output["nbtag"].fill(dataset=dataset, category=category, nbtag=nbtag, weight=sample_weight*np.ones(len(nbtag)))
            #njets = good_jets.counts.flatten()
            #output["njets"].fill(dataset=dataset, category=category, njets=njets, weight=sample_weight*np.ones(len(njets)))

        return self.output

    def postprocess(self, accumulator):
        return accumulator
