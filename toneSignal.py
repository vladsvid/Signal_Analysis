# -*- coding: utf-8 -*-
import numpy as np
import os


signal_params_2400 = {
    'Fclk': 2.4e+6,
    'zero_one_convert':True,
    'bw': 'npm',
    'cut_first_bin': True,
    'noiseOnly': False,
    'aweight': False,
    'window': 'hann',  
    'ampCor': True
    }



class PdmSignalPreparation:
    def __init__(self, path, params):
        self.path = path
        self.params = params

        self.pdmData = np.loadtxt(path, dtype=float)
        self.filename = os.path.basename(self.path)

        if self.params['zero_one_convert']:
            self.pdmData = self.pdmData*2 - 1
            
        if self.params['cut_first_bin']:
            onebin = int(len(self.pdmData)/11)
            self.pdmData = self.pdmData[onebin:]
            


    

