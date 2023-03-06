# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 15:28:01 2020

@author: VSviderskij
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.interpolate import interp1d

class FFT_ANALYSIS:
    bandwidth = {'npm': (20e+0,20e+3),
                 'lpm': (20e+0, 8e+3)}
    
    def __init__(self, path, params):
        import numpy as np
        import os
        self.path = path
        
        self.file_stat = os.stat(self.path)
        print('File size:', round(self.file_stat.st_size/(1024*1024), 2), 'Mb')
        
        self.filename = os.path.basename(self.path)
        
        self.Fclk = params['Fclk']
        self.bw = params['bw']
        self.noiseOnly = params['noiseOnly']
        self.aweight = params['aweight']
        self.in_volts = params['in_volts']
        self.half_spectrum = params['half_spectrum']
        
        
        
        self.inp = np.loadtxt(self.path, dtype=float)
        
        if not self.in_volts:
            self.inp = np.power(10,self.inp/20)
            


        

        
        if self.half_spectrum:
            self.N = int(len(self.inp) * 2)
        else:
            self.N = int(len(self.inp))
            
        self.inp = self.inp.astype(float)
        
        self.dc_level = self.dc_level()
        
        self.f_low = FFT_ANALYSIS.bandwidth[self.bw][0]
        self.f_high = FFT_ANALYSIS.bandwidth[self.bw][1]
        
        self.fb_resolution = round(self.Fclk/self.N,2)
        
        self.ind_high = int(np.round(self.f_high*self.N/self.Fclk))       
        self.ind_low = int(np.round(self.f_low*self.N/self.Fclk))  

        if self.ind_low <= 0:
            self.ind_low = 2

        if self.f_high > (self.Fclk/2):
            self.f_high=self.Fclk/2; # the upper frequency cannot be larger than half the clock frequency
        
        if self.ind_high < self.ind_low:
            self.ind_high = self.ind_low
            
        self.signalDescription()
            
    def signalDescription(self):
        print('=== Data description ===')
        print('Number of samples:', len(self.inp))   
        if len(self.inp) % 2 == 1:
            self.inp = np.append(self.inp, 0)
            print('Size of the input vector is not even, an zero is added at the end of vector')    
        print('Low bandwidth:', int(self.f_low), 'Hz')
        print('High bandwidth:', int(self.f_high), 'Hz')
        print('Freq bin resolution:', self.fb_resolution, 'Hz')

        if self.noiseOnly:
            print("Noise only analyzed, the tone won't be cut from measurements")              
        
            
    def idx_to_freq(self, idx, N, Fclk):
        return np.round(idx*Fclk/N)

    def freq_to_idx(self, freq, N, Fclk):
        return np.round(freq*N/Fclk)            
        
    def dc_level(self):
        return np.log10(np.abs(np.mean(self.inp)))*20
    
    def dbfs(self, float):
        return 20*np.log10(np.abs(float))        
    
    def calculate_tone(self, plot=False):
          
        temp = self.dbfs(self.inp)

        y1 = temp[self.ind_low: self.ind_high] 
        
        f_idx = np.argmax(y1)
        f_idx = f_idx + self.ind_low
        
        f_signal = (f_idx*self.Fclk)/self.N
        
        m = self.inp[f_idx-1:f_idx+2]
        m = np.linalg.norm(m)     
        a_f = self.dbfs(m)
        a_f = round(a_f,2)
        
        plot_params = dict()
        plot_params['a_f'] = a_f
        plot_params['f_signal'] = f_signal
        plot_params['signal_to_plot'] = temp
        plot_params['test_name'] = 'tone'
        
        if plot:
            
            self.plot(plot_params)
      
        return {'f_idx': f_idx,
                'f_signal': f_signal,
                'tone_volt': m,
                'tone_dbfs': a_f,
                'temp_arr': temp
                }
        
    def plot(self, params):
        
        arr = params['signal_to_plot']
        test_name = params['test_name']
            
        plt.figure(figsize=(800/100, 400/100), dpi=100)
        
        plt.xscale('symlog')
        plt.ylabel('Power [dBFs]')
        plt.xlabel('Frequency [Hz]')
        plt.axis([0, self.Fclk/2, -200, 0])
        plt.title("{} plot.\n File name: {}".format(params['test_name'].upper(), self.filename))
        plt.grid(True)   
        label = None
        
        text = ''
        
        if test_name == 'tone':
            text = '''Signal Descrption:
Fclk     {}kHz
Signal {:.4}dBFs
Signal f  {}kHz'''.format(self.Fclk/1000,params['a_f'],params['f_signal']/1000)
        
        elif test_name == 'thd':
            text = '''Signal Descrption:
Fclk                {}kHz
Fundamental Signal  {:.4}dBFs
Signal freq         {}kHz   
THD                 {}%'''.format(self.Fclk/1000,params['p_f'],params['f_signal']/1000,params['thd'])            
        
        elif test_name == 'noise':
            
            text = '''Signal Descrption:
Fclk           {:4}kHz
Noise          {:.4}dBFS
SNR            {} (dB)
A-weighting    {}'''.format(self.Fclk/1000,params['noise_dbfs'],params['snr'], self.aweight)
        
        else:
            raise Exception('No test name, or wrong spelling')
        
        
        freqs = np.linspace(0,int(self.Fclk/2),num = int(self.N/2))
        
        
        plt.plot(freqs, arr[0:int(self.N/2)], 'b-', label = label)
        
        plt.text(0.3, -5, text, verticalalignment='top',\
                 horizontalalignment='left', \
                bbox=dict(facecolor='white', \
                edgecolor='black', fill=True))
        
        
    def calculate_THD(self, plot=False):
        
        f_tone = self.calculate_tone(plot=False)
        p_f = f_tone['tone_volt']
        f_idx = f_tone['f_idx']
        f_signal = f_tone['f_signal']
        
        idx = f_idx*2 # to start from second harmonics
            
        p_harmonics = 0
        print('index',idx)
        print(f_idx)
        print(p_f)
        
        while idx <= self.ind_high:
            m = self.inp[idx-1:idx+2]          
            m = np.linalg.norm(m)
            p_harmonics = p_harmonics + np.square(m)
            idx = idx+f_idx
            print(m)
           
        thd = 100*np.sqrt(p_harmonics)/p_f
        thd = round(thd,2)
        p_f = self.dbfs(p_f)
        
        plot_params = dict()
        plot_params['p_f'] = p_f
        plot_params['f_signal'] = f_signal
        plot_params['signal_to_plot'] = self.dbfs(self.inp)
        plot_params['test_name'] = 'thd'  
        plot_params['thd'] = thd 
        
        if plot:
            self.plot(plot_params)
            
        return thd
    
    def calculate_noise(self, plot=False):
        
        if not self.noiseOnly:
            f_tone = self.calculate_tone(plot=False)
            f_idx = f_tone['f_idx']


        temp_dbfs = self.dbfs(self.inp[0:int(self.N/2)])

        if self.aweight:
            
            dds = self.inp[0:int(self.N/2)]
            dds = self._aweight(dds)
            temp_dbfs = self.dbfs(dds)            
        
        if not self.noiseOnly:
            print ("The Fundamental excluded from noise measurement")
            if f_idx < 3:
                #temp_dbfs[0:int(f_idx)+4] = -400;        # removing tone/signal
                x = np.arange(0,f_idx+4)
                init_y = [temp_dbfs[0],temp_dbfs[f_idx+4]]
                init_x = [0,f_idx+4]
                interp = np.interp(x,init_x,init_y)
                temp_dbfs[0:f_idx+4] = interp
            else:
               #temp_dbfs[f_idx-3:f_idx+4] = -400;      # removing tone/signal
                
                x = np.arange(f_idx-3,f_idx+4)
                init_y = [temp_dbfs[f_idx-3],temp_dbfs[f_idx+4]]
                init_x = [f_idx-3,f_idx+4]
                interp = np.interp(x,init_x,init_y)
                temp_dbfs[f_idx-3:f_idx+4] = interp

        else:
            print ("The Fundamental NOT excluded from noise measurement")
            
        t_dbfs = temp_dbfs[self.ind_low:self.ind_high]
                        
        noise_volt = np.linalg.norm(np.power(10,t_dbfs/20))
        
        noise_dbfs = self.dbfs(noise_volt)
        
        if self.noiseOnly:
            snr = 'No signal'
        else:
            snr = round(f_tone['tone_dbfs'] - noise_dbfs,2)
        
        plot_params = dict()
        plot_params['noise_dbfs'] = noise_dbfs
        plot_params['signal_to_plot'] = temp_dbfs
        plot_params['test_name'] = 'noise'  
        plot_params['snr'] = snr
        
        if plot:
            self.plot(plot_params)

        return {'noise_dbfs': noise_dbfs,
                'snr': snr,
                'noise_arr_dbfs': temp_dbfs,
                }    
    

    def _aweight(self, sig_arr):
        
        k = 1.8719e+8
        _sig_arr = sig_arr.copy()
        _sig_arr[0] = _sig_arr[0]*1e-11
        for i in range(1, len(sig_arr)):
            
            f = i* self.fb_resolution
            f4 = f**4
            m1 = k*f4
            n1 = ((20.598997)**2 + f**2)**(-1)
            n2 = ((107.65265)**2 + f**2)**(-0.5)
            n3 = ((737.86223)**2 + f**2)**(-0.5)
            n4 = ((12194.217)**2 + f**2)**(-1)
            result = abs(m1*n1*n2*n3*n4)
            _sig_arr[i] = _sig_arr[i]*result
           
        return _sig_arr
        
params_noise_2400 = {
    'Fclk': 2.4e+6,
     'bw': 'npm',
     'noiseOnly': True,
     'aweight': True,
     'in_volts':True,
     'half_spectrum': True
    }

params_noise_800 = {
    'Fclk': 0.8e+6,
     'bw': 'lpm',
     'noiseOnly': True,
     'aweight': True,
     'in_volts':True,
     'half_spectrum': True
    }

params_signal_2400 = {
    'Fclk': 2.4e+6,
     'bw': 'npm',
     'noiseOnly': False,
     'aweight': False,
     'in_volts':True,
     'half_spectrum': True
    }

params_signal_800 = {
    'Fclk': 0.8e+6,
     'bw': 'npm',
     'noiseOnly': False,
     'aweight': False,
     'in_volts':True,
     'half_spectrum': True
    }
        
file = r"C:\Users\VSviderskij\Desktop\DEVs\PROJECTS\MacDuff\TESTS DEBUG\SetOfTests\SetOfTests2\SetOfTests2\THD120__800KHz_FFTAbs_012721.txt"
lala = FFT_ANALYSIS(file, params_signal_800)
#%%
noise = lala.calculate_noise(plot=True)
#%%
tone = lala.calculate_tone(plot=True)
#%%
thd = lala.calculate_THD(plot=True)


#%%
def dbfs(float):
        return 20*np.log10(np.abs(float))  


data = np.loadtxt(file)

dataDb = dbfs(data)








