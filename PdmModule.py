# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 09:38:09 2021

@author: VSviderskij
"""
import numpy as np
from abc import ABC, abstractmethod
import pathlib
params = {
'file_path': r"C:\Users\VSviderskij\Desktop\DEVs\PROJECTS\Miltonduff\test program\12_05_2022\DATAPDM\DATAPDM\THD94_NMP1_311mVp_PDM_5122022.txt",
'Fclk': 3072000,
'BWLow': 20,
'BWHigh': 20000,
'a-weight': True,
'windowing': 'hanning',
'correction': 'power'
}
class ReadData(ABC):

    @abstractmethod
    def read(self):
        pass
    
    def data_description(self):
        pass

class PdmData(ReadData):
    def __init__(self, config):
        self.config = config
        self.data = None
        self.zeroOneConvert = config.get('convert_zero_one', 1)
        self.cutBin = config.get('cut_first_bin', 0)        
        self.N = None
        self.read()
        self.bin_resolution = round(self.config['Fclk']/self.N,2)

    def data_description(self):
        print('File name: {}'.format(pathlib.Path(self.config['file_path']).name))
        print('Samples rate: {}'.format(len(self.data)))
        print('Clock Freq: {}'.format(self.config['Fclk']))
        print('Low bandwidth: {}'.format(self.config['BWLow']))
        print('High bandwidth: {}'.format(self.config['BWHigh']))
        if self.zeroOneConvert:
            print('Signal converted from 0/1 to -1/1')
        print('Bin resolution: {}'.format(self.bin_resolution))
        
        
        
    def read(self):
        self.read_data()

        if self.zeroOneConvert:
            self.data = self.data*2 - 1
            self.N = len(self.data)
        if self.cutBin:
            self.cut_bin()
            self.N = len(self.data)
            
    def cut_bin(self):
        onebin = int(len(self.data)/11)
        self.data = self.data[onebin:]

    def read_data(self):

        with open(self.config['file_path']) as file:
            self.data = file.readlines()
            self.data = [ float(x) for x in self.data]
            self.data = np.array(self.data)

     

lala = PdmData(params)
lala.data_description()


class FFT:
    compens_factor = {'hanning': { 'power': 1,
                                   'amplitude': 1},
                      'hamming': {'power': 1,
                                  'amplitude': 1}
                      }
        
    def __init__(self, params):
        self.arr = params['data_array']
        self.correction = params['correction']
        self.windowing = params['window']
        self.fft_signal = None
        
    def _apply_window(self):
        if self.windowing == 'hanning':
            self.apply_hanning()
        elif self.windowing == 'hamming':
            self.apply_hamming()
        else:
            pass
        
    def _apply_correction(self):
        self.arr = self.arr*(FFT.compens_factor[self.windowing][self.correction])

    def _calculate_fft(self):
        y0 = np.fft.fft(self.arr)
        y0 = abs(y0)
        s0 = y0/len(y0)
        self.fft_signal = s0*2 # full spectrum compensation
        self.fft_signal = self.fft_signal[0:int(len(self.fft_signal)/2)]
        
    def fft(self):
        self._apply_window()
        self._apply_correction()
        self._calculate_fft()
    
    
    def apply_hanning(self):
        N = len(self.arr)
        win = np.hanning(N)
        inp = np.multiply(self.arr, win)
        return inp
    
    def apply_hamming(self):
        N = len(self.arr)
        win = np.hamming(N)
        inp = np.multiply(self.arr, win)
        return inp         
        
class Signal:
    
    def to_dbfs(self, float):
        return 20*np.log10(np.abs(float))    
    
    def from_dbfs(self, float):
        return 10**(float/20)
    
     
class Tone(Signal):
    def __init__(self, params):
        self.arr = params['fft_array']
        self.ind_low = params['ind_low']
        self.ind_low = params['ind_high']
        self.Fclk = params['fclk']
        self.correction = params['correction']
        self.in_dbfs = params['is_in_dbfs']
        
        
    def calculate_tone(self):
        if self.in_dbfs:
            temp_volt = self.from_dbfs(self.arr)
            temp_dbfs = self.arr
        else:
            temp_volt = self.arr
            temp_dbfs = self.to_dbfs(self.arr)
        

        
        y1 = temp_dbfs[self.ind_low: self.ind_high] 
        
        f_idx = np.argmax(y1)
        f_idx = f_idx + self.ind_low
        
        f_signal = (f_idx*self.Fclk)/len(self.arr)
        
        if self.correction == 'amplitude':
            m = temp_volt[f_idx]
        elif self.correction == 'power':
            m = temp_volt[f_idx-1:f_idx+2]
            m = np.linalg.norm(m)     
        
        a_f = self.dbfs(m)
        a_f = round(a_f,3)


      
        return {'f_idx': f_idx,
                'f_signal': f_signal,
                'tone_volt': m,
                'tone_dbfs': a_f,
                'array_dbfs': temp_dbfs,
                'array_ampl': temp_volt
                }        
    
class THD(Signal):
    
    def __init__(self, params):
        self.params = params
        self.arr = params['fft_array']
        self.ind_low = params['ind_low']
        self.ind_low = params['ind_high']
        self.Fclk = params['fclk']
        self.correction = params['correction']
        self.in_dbfs = params['is_in_dbfs']        
        
    def calculate_thd(self):
        tone = Tone(self.params)
        f_tone = tone.calculate_tone()
        
        p_f = f_tone['tone_volt']
        f_idx = f_tone['f_idx']
        f_signal = f_tone['f_signal']
        temp_volt = f_tone['array_ampl']
        
        idx = f_idx*2 # to start from second harmonics
        
        p_harmonics = 0
        
        while idx <= self.ind_high:
            
            if self.correction == 'amplitude':
                m = temp_volt[f_idx]
            elif self.correction == 'power':
                m = temp_volt[f_idx-1:f_idx+2]
                m = np.linalg.norm(m)    
            
            p_harmonics = p_harmonics + np.square(m)
            idx = idx+f_idx
           
        thd = 100*np.sqrt(p_harmonics)/p_f
        thd = round(thd,4)
       
        
        results = dict()
        results['p_f'] = f_tone['tone_dbfs']
        results['f_signal'] = f_signal
        results['signal_to_plot'] = temp_volt
        results['test_name'] = 'thd'    
        results['thd'] = thd
        
        return results    
    
class Noise(Signal):
    def __init__(self, params):
        self.params = params
        self.arr = params['fft_array']
        self.ind_low = params['ind_low']
        self.ind_low = params['ind_high']
        self.Fclk = params['fclk']
        self.correction = params['correction']
        self.in_dbfs = params['is_in_dbfs']    
        
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
        
    def calculate_noise(self):

            
        if not self.noiseOnly:
            tone = Tone(self.params)
            f_idx = tone.calculate_tone()['f_idx']
         
            
        if self.in_dbfs:
            temp_volt = self.from_dbfs(self.arr)
            temp_dbfs = self.arr
        else:
            temp_volt = self.arr
            temp_dbfs = self.to_dbfs(self.arr)
        

        if self.aweight:
            dds = self._aweight(temp_volt)
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
     
        # if self.noiseOnly:
        #     snr = 'No signal'
        # else:
        #     snr = round(f_tone['tone_dbfs'] - noise_dbfs,2)        

        results = dict()
        results['noise_dbfs'] = noise_dbfs
        results['signal_to_plot'] = temp_dbfs
        results['test_name'] = 'noise'  
 
        return {'noise_dbfs': noise_dbfs,
                'snr': snr,
                'noise_arr_volt': temp_volt,
                'noise_arr_dbfs': temp_dbfs
                }  