# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:34:51 2021

@author: VSviderskij
"""
import numpy as np
import os

class SignalBase:
    def __init__(self, path, Fclk, bwLow, bwHigh):
        self.path = path
        self.Fclk = Fclk        
        self.loadData()
        self.calcBinResolution()
        self.highBW = bwHigh
        self.lowBW = bwLow
        self._getBWIndexes()
        
        self.win = None
        self.distWinType = None
        self.distWinComp = 1
        
        self.fftApplied = False
        self.winApplied = False

        
        #self.getFileDescription()
        
    def calcBinResolution(self):        
        self.resolution = round(self.Fclk/self._arrLen(),2)
       
    def _arrLen(self):
        return len(self.data)
    
    def _getBWIndexes(self):
        self.lowBWInd = int(np.round(self.lowBW*self._arrLen()/self.Fclk))
        self.highBWInd = int(np.round(self.highBW*self._arrLen()/self.Fclk))  
        self.highBWInd = self.highBWInd + 1        
        
        if self.lowBWInd <= 0:
            self.lowBWInd = 2

        if self.highBWInd > (self.Fclk/2):
            self.highBWInd=self.Fclk/2; # the upper frequency cannot be larger than half the clock frequency
        
        if self.highBWInd < self.lowBWInd:
            self.highBWInd = self.lowBWInd

    def loadData(self):
        self.file_stat = os.stat(self.path)
        self.file_size = round(self.file_stat.st_size/(1024*1024), 2)
        
        print('=== File description ===')
        print('File size:', self.file_size, 'Mb')
        
        if self.file_size > 5:
            print('It may take some time to load')
            
        self.data = np.loadtxt(self.path, dtype=int)   
        
                
    def showParameters(self):
        print()
        print('=== Arguments ===')
        print('Number of samples:', self._arrLen()) 
        print('Fclk:', self.Fclk, 'Hz')
        print('Low BW:', self.lowBW, 'Hz' )
        print('High BW:', self.highBW, 'Hz' )
        print('Resolution:', self.resolution)
        print()
        print('Windowing:', self.win)
        print('Windowing Distortion Type:', self.distWinType)
        print()
        print('Piece of data:', self.data[:5])
        print('Comp factor:', self.distWinComp)

    def cutFirstBin(self):
            onebin = int(len(self.data)/11)
            self.data = self.data[onebin:]
            print('First bin cut, number of samples:', self._arrLen())    
            self.calcBinResolution()
            print('Resolution updated:', self.resolution)
        
    def setBW(self, low, high):
        self.lowBW = low
        self.highBW = high
    
    def convertZeroOne(self):
        self.data = self.data*2 - 1
        print('1 and 0 converted to 1 and -1')
    
    def setClock(self, clk):
        self.Fclk = clk
        
    def setWindowingType(self, win):
        self.win = win
    
    def setDistWinType(self, distWinType):
        self.distWinType = distWinType

    def applyWindowing(self):
        if None in (self.win, self.distWinType):
            raise Exception('Cannot calculate window array without required arguments')
    
        ww = Windowing(self._arrLen(), self.win, self.distWinType)
        winArr = ww.getWinArr()
        
        self.distWinComp = ww.getCompenssation()
        self.data = np.multiply(winArr, self.data)
        self.winApplied = True
        
    def dbfs(self, float):
        return 20*np.log10(np.abs(float))            
               

class Windowing:
    compFactors = {'hann':{'energy':1.63,'amplitude':2},
                   'hamm':{'energy':1.85,'amplitude':1.59}
                   }
    
    def __init__(self, arr_len, win, distType):
        self.arrLen = arr_len
        self.win = win
        self.distType = distType
    
    def hanning(self):
        return np.hanning(self.arrLen)
        
    
    def hamming(self):
        return np.hamming(self.arrLen)
    
    def getWinArr(self):
        
        if self.win == 'hann':
            return self.hanning()
        elif self.win == 'hamm':
            return self.hamming()
        else:
            raise('Unknown windowing applied')
    
    def getCompenssation(self):
        return Windowing.compFactors[self.win][self.distType]
    
                
class Tone(SignalBase):
    
    def calcFFT(self, halfSpectrum = True):
        fftComplex = np.fft.fft(self.data)
        fftAbs = abs(fftComplex)
        fftNorm= fftAbs/len(fftAbs)
        
        if halfSpectrum:
            halfFFTLen = int(len(fftNorm)/2)
            halfFFT = fftNorm[:halfFFTLen] # get firt half of spectrum
            halfFFT = halfFFT*2 # compensate the energy of second half spectrum
            self.data = halfFFT
        else:
            self.data = fftNorm
        
        self.fftApplied = True
            
    def calcSens(self):
        if not self.fftApplied:
            raise Exception('FFT is not applied, please do so')
        
        tempFFT = self.data[self.lowBWInd: self.highBWInd]
        self.tone_ind = np.argmax(tempFFT)
        self.tone_ind = self.tone_ind + self.lowBWInd
        
        self.tone_freq = (self.tone_ind*self.Fclk)/self._arrLen()
        
        m = self._calcTone(self.tone_ind)
        
        self.sensDbfs = round(self.dbfs(m), 3)
        self.sensVolts = round(m,3)
        
    def calcTHD(self):
        self.calcSens()
                
        ind_thd = self.tone_ind*2 # to start from second harmonics
        p_harmonics = 0
        while ind_thd < self.highBWInd:
            
            m = self._calcTone(ind_thd)
            print(m)
            #m = m*self.distWinComp
        
            p_harmonics = p_harmonics + np.square(m)
            ind_thd = ind_thd + self.tone_ind
            
        self.thd = 100*np.sqrt(p_harmonics)/self.sensVolts
        self.thd = round(self.thd,2)        
    
    def _calcTone(self, value:int):
        
        if self.distWinType == 'energy':
            m = self.data[value-1:value+2]
            m = np.linalg.norm(m) 
        elif self.distWinType == 'amplitude' or self.distWinType is None:
            m = self.data[value]    
        else:
            raise Exception('Unknown signal distortion type')      
        m = m*self.distWinComp
        return m
        

class Noise(Tone):
    
    def __init__(self, path, Fclk, bwLow, bwHigh, excludeTone = False):
        super().__init__(path, Fclk, bwLow, bwHigh)
        self.aweight = False
        
        if excludeTone:
            self.calcSens()
        
    
    def calcNoise(self):
        pass
        
    
    
    
    
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
    
    
    
           
        
        
        
        
        
        
        
       
        
        
        
        
        
        
        
        
        
noise_path = r"C:\Users\VSviderskij\Desktop\DEVs\PROJECTS\MacDuff\TESTS DEBUG\SetOfTests\SetOfTests\Noise_800KHz_PDM_012621.txt"
noise = Noise(noise_path, 2.4e+6, 20, 2e+4)
        
#%%

                    

path = r"C:\Users\VSviderskij\Desktop\DEVs\PROJECTS\MacDuff\TESTS DEBUG\SetOfTests\SetOfTests2\SetOfTests2\THD120__2p4MHz_PDM_012721.txt"
tone = Tone(path, 2.4e+6, 20, 2e+4)
tone.cutFirstBin()
tone.showParameters()
#%%

tone.setBW(20,2e+4)
tone.convertZeroOne()
tone.setClock(2.4e+6)
tone.calcBinResolution()
#tone.setWindowingType('hann')
#tone.setDistWinType('energy')
#tone.applyWindowing()
tone.showParameters()
tone.calcFFT(halfSpectrum=True)
foo = tone.data
print(tone.lowBWInd)
print(tone.highBWInd)
tone.calcSens()
tone.calcTHD()
print(tone.thd)
# print(tone.f_ind)
# print(tone.f_signal)
# print(tone.sensDbfs)

    
    # def dbfs(self, ):
    #     pass
    
    # def getTHDCalculation():
    #     pass
    
    # def getNoiseCalculation():
    #     pass
    
#%%
  
        
        
    #     if len(self.data) % 2 == 1:
    #         self.data = np.append(self.data, 0)
    #         print('Size of the dataut vector is not even, an zero is added at the end of vector')
        
    #     if self.zeroOne:
    #         self.data = self.data*2 - 1
    #         print('1 and 0 converted to 1 and -1')
            
            
    #     print('Low bandwidth:', int(self.f_low), 'Hz')
    #     print('High bandwidth:', int(self.f_high), 'Hz')
    #     print('Freq bin resolution:', self.resolution, 'Hz')    