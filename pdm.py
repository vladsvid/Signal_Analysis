import numpy as np

class PDM_ANALYSIS:
    bandwidth = {'npm': (20e+0,20e+3),
                 'npm20k': (50e+0,24e+3),
                 'lpm': (50e+0, 8e+3)}
    
    compens_factor = {'hann': 3.26,
                      'hannAmp': 4,
                      'hamm': 3.18,
                      None : 1}
    
    def __init__(self, path, params):
        import numpy as np
        import os
        self.path = path
        
        self.file_stat = os.stat(self.path)

        
        
        self.zeroOne = params['zero_one_convert']
        self.cutFirstBin = params['cut_first_bin']
        self.Fclk = params['Fclk']
        self.bw = params['bw']
        self.noiseOnly = params['noiseOnly']
        self.aweight = params['aweight']
        self.window = params['window']
        self.ampCorr = params['ampCor']
        
        self.filename = os.path.basename(self.path)
        self.inp = np.loadtxt(self.path, dtype=float)
        
        try:
            self.wind_compens = PDM_ANALYSIS.compens_factor[self.window]
        except KeyError as e:
            raise Exception('Wrong window name',e)
            
        if self.cutFirstBin:
            self.onebin = int(len(self.inp)/11)
            self.inp = self.inp[self.onebin:]
            print('First bin cut, number of samples:', len(self.inp))                 
        
        self.N = len(self.inp)
        self.inp = self.inp.astype(float)
        self.dc_level = self.dc_level()
        
        self.f_low = PDM_ANALYSIS.bandwidth[self.bw][0]
        self.f_high = PDM_ANALYSIS.bandwidth[self.bw][1]
        
        self.fb_resolution = round(self.Fclk/self.N,2)
        self.ind_high = int(np.round(self.f_high*self.N/self.Fclk))
        #self.ind_high = self.ind_high + 1        
        self.ind_low = int(np.round(self.f_low*self.N/self.Fclk))  
        
        if self.ind_low <= 0:
            self.ind_low = 2

        if self.f_high > (self.Fclk/2):
            self.f_high=self.Fclk/2; # the upper frequency cannot be larger than half the clock frequency
        
        if self.ind_high < self.ind_low:
            self.ind_high = self.ind_low
            

            
        
    
    def signalDescription(self):
        print('=== Data description ===')
        print('Number of samples:', len(self.inp))
        if len(self.inp) % 2 == 1:
            self.inp = np.append(self.inp, 0)
            print('Size of the input vector is not even, an zero is added at the end of vector')
        
        if self.zeroOne:
            self.inp = self.inp*2 - 1
            print('1 and 0 converted to 1 and -1')
            
            
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
    
    def apply_hann(self, arr):
        N = len(arr)
        win = np.hanning(N)
        inp = np.multiply(arr, win)
        return inp
    
    def apply_hamm(self, arr):
        N = len(arr)
        win = np.hamming(N)
        inp = np.multiply(arr, win)
        return inp
        
    def calculate_fft(self, arr):
        if self.window == 'hann':
            arr = self.apply_hann(arr)
        elif self.window == 'hamm':
            arr = self.apply_hamm(arr)
        else:
            arr = arr
            
        y0 = np.fft.fft(arr)
        y0 = abs(y0)
        s0 = y0/len(y0)
        #Compensating
        s0 = s0*self.wind_compens
        return s0
        
    def dbfs(self, float):
        return 20*np.log10(np.abs(float))        
    
    def calculate_tone(self, params=None, plot=False):
        
        if params == None:
            inp = self.inp
            ind_low = self.ind_low
            ind_high = self.ind_high 
        else:
            inp = params['inp']
            ind_low = params['ind_low']
            ind_high = params['ind_high'] 
        
        temp_fft = self.calculate_fft(inp)
        temp = self.dbfs(temp_fft)
        
        y1 = temp[ind_low: ind_high] 
        
        f_idx = np.argmax(y1)
        f_idx = f_idx + ind_low
        
        f_signal = (f_idx*self.Fclk)/len(inp)
        
        if self.ampCorr:
            m = temp_fft[f_idx]
      
        else:
            m = temp_fft[f_idx-1:f_idx+2]
            m = np.linalg.norm(m)     
        
        a_f = self.dbfs(m)
        a_f = round(a_f,3)
        
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
        
  
    def calculate_THD(self, params = None, plot=False):
        
        if params == None:
            inp = self.inp
            ind_high = self.ind_high
        else: 
            inp = params['inp']
            ind_high = params['ind_high']
        
        f_tone = self.calculate_tone(params=params, plot=False)
        p_f = f_tone['tone_volt']
        f_idx = f_tone['f_idx']
        f_signal = f_tone['f_signal']
        temp = f_tone['temp_arr']
        
        idx = f_idx*2 # to start from second harmonics
        
        s = self.calculate_fft(inp)
        
        p_harmonics = 0
        while idx <= ind_high:
            
            if self.ampCorr:
                m = s[idx]
                print(idx, m)
            else:
                m = s[idx-1:idx+2]
                m = np.linalg.norm(m)
            
            p_harmonics = p_harmonics + np.square(m)
            idx = idx+f_idx
           
        thd = 100*np.sqrt(p_harmonics)/p_f
        thd = round(thd,4)
        p_f = self.dbfs(p_f)
        
        plot_params = dict()
        plot_params['p_f'] = p_f
        plot_params['f_signal'] = f_signal
        plot_params['signal_to_plot'] = temp
        plot_params['test_name'] = 'thd'    
        plot_params['thd'] = thd
        
        if plot:
            self.plot(plot_params)
            
        return thd
    
    def calculate_noise(self, params=None, plot=False):
        if params == None:
            inp = self.inp
            ind_low = self.ind_low
            ind_high = self.ind_high
        else: 
            inp = params['inp']
            ind_low =  params['ind_low']
            ind_high = params['ind_high']
            
        if not self.noiseOnly:
            f_tone = self.calculate_tone(plot=False)
            f_idx = f_tone['f_idx']            
            
        temp_volt = self.calculate_fft(inp)
        temp_dbfs = self.dbfs(temp_volt[0:int(len(inp)/2)])

        if self.aweight:
            
            dds = temp_volt[0:int(len(inp)/2)]
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
            
        t_dbfs = temp_dbfs[ind_low:ind_high]
                        
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
        
        if plot:
            self.plot(plot_params)

        return {'noise_dbfs': noise_dbfs,
                'snr': snr,
                'noise_arr_volt': temp_volt,
                'noise_arr_dbfs': temp_dbfs
                }    
    
    def signal_sweep(self, signal, freq_bin_res=100, step=1e-3, plot=False, plot_chunk=None):
        
        step_in_samples = int(self.Fclk*step)
        wind_in_samples = int(self.Fclk/freq_bin_res)
        
        if wind_in_samples >= len(self.inp):
            raise Exception('Sweep window larger then collected signal due to high resolution')
         
        ind_low = 0
        ind_high = wind_in_samples
        count = 0
        
        bw_low = int(np.round(self.f_low*wind_in_samples/self.Fclk))  
        bw_high = int(np.round(self.f_high*wind_in_samples/self.Fclk))
        
        if bw_low <= 0:
            bw_low = 2

        if bw_high < bw_low:
            bw_high = bw_low
        
        params = {'inp': None,
                  'ind_low': None,
                  'ind_high': None
                  }
    
        signal_arr = list()
        
        print(step_in_samples, wind_in_samples, ind_low, ind_high, len(self.inp))
        wind_details = dict()
        while ind_high < len(self.inp)+1:
            
            temp_wind = self.inp[ind_low:ind_high+1]

            params['inp'] = temp_wind
            params['ind_low'] = bw_low
            params['ind_high'] = bw_high
            y_label = ''
            if signal == 'thd':
                signal_calc = self.calculate_THD(params=params)
                y_label = 'THD in %'
            elif signal == 'tone':
                signal_calc = self.calculate_tone(params=params)
                signal_calc = signal_calc['tone_dbfs']
                y_label = 'dBFS'
            elif signal == 'noise':
                signal_calc = self.calculate_noise(params=params)
                signal_calc = signal_calc['noise_dbfs']
                y_label = 'dBFS'
                
            
            signal_arr.append(signal_calc)
            print(count, ind_low, ind_high, ind_high-ind_low, signal_calc)
            wind_details[count] = (ind_low,ind_high)
            
            ind_low = ind_low + step_in_samples
            ind_high = ind_low + wind_in_samples
            count +=1
            
        if plot:
            sigma = np.std(signal_arr)
            mean = np.mean(signal_arr)
            print('Sigma', sigma)
            print('mean', mean)
            text = "Sigma    {}\nAverage    {}".format(sigma, mean)
            
            
            x = np.linspace(0, len(signal_arr), num=len(signal_arr))
            plt.figure()
            plt.plot(x,signal_arr)
            plt.ylabel(y_label)
            plt.xlabel('steps ({}sec or {} samples per step)'.format(step, step_in_samples))
            plt.title('Signal sweep, FB resolution {} Hz.Sigma {:.5}. Avg {:.5}. File {}'.format(res, sigma, mean, self.filename))
            plt.text(0.3, -5, text, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='black', fill=True))
            plt.grid()
            
        if plot_chunk is not None:
            _low = wind_details[plot_chunk][0]
            _high = wind_details[plot_chunk][1]
            data = self.inp[_low:_high]
            print(len(data))
            
            params['inp'] = data
            params['ind_low'] = bw_low
            params['ind_high'] = bw_high

            if signal == 'thd':
                signal_calc = self.calculate_THD(params=params, plot=True)
                y_label = 'THD in %'
            elif signal == 'tone':
                signal_calc = self.calculate_tone(params=params,plot=True)
                signal_calc = signal_calc['tone_dbfs']
                y_label = 'dBFS'
            elif signal == 'noise':
                signal_calc = self.calculate_noise(params=params,plot=True)
                signal_calc = signal_calc['noise_dbfs']
                y_label = 'dBFS'
                
                
        return signal_arr, signal_calc
    

    # def _aweight(self, numSamples, sample_rate):
    
    #     k = 1.8719e+8
    #     aw_filter = list()
    #     aw_filter.append(1e-11)
    #     for i in range(1, numSamples):
            
    #         f = i* sample_rate / numSamples
    #         f4 = f**4
    #         m1 = k*f4
    #         n1 = ((20.598997)**2 + f**2)**(-1)
    #         n2 = ((107.65265)**2 + f**2)**(-0.5)
    #         n3 = ((737.86223)**2 + f**2)**(-0.5)
    #         n4 = ((12194.217)**2 + f**2)**(-1)
    #         result = abs(m1*n1*n2*n3*n4)
    #         aw_filter.append(result)
    #     aw_f = np.array(aw_filter)
    #     return aw_f
    
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
        


def main(file, sig):
    lala = PDM_ANALYSIS(file, sig)
    tone = lala.calculate_tone()
    print(str(tone))

import sys
import json
if __name__ == '__main__':
    

    sig = json.loads(sys.argv[2])
    # {
    # 'Fclk': int(sys.argv[1]),
    # 'zero_one_convert':int(sys.argv[2]),
    # 'bw': sys.argv[3],
    # 'cut_first_bin': int(sys.argv[4]),
    # 'noiseOnly': int(sys.argv[5]),
    # 'aweight': int(sys.argv[6]),
    # 'window': sys.argv[7],
    # 'ampCor': int(sys.argv[8])
    # }

    main(sys.argv[1], sig)

