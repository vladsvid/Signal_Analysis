# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 09:20:27 2021

@author: VSviderskij
"""

import numpy as np

path = r"C:\Users\VSviderskij\Desktop\DEVs\PROJECTS\MacDuff\TESTS DEBUG\10s_PDM Data\10s_PDM Data\Noise_800KHz_10sPDM_012821.txt"
# with open(path, 'r') as f:
#     data = f.readlines()

data = np.loadtxt(path)
 #%%
 
#line = data[0].split(',')
line = data[:800000]
line = np.array(line).astype(int)

#%%
np.savetxt(r'C:\Users\VSviderskij\Desktop\DEVs\PROJECTS\MacDuff\TESTS DEBUG\Noise debug\2020-05-11_10-02-05\dut1_800_1s.txt', line, fmt='%i')


