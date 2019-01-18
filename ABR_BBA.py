
'''
                             AI-Trans Competition
*****************************************************************************
*  In this version of algorithm, we use improved A3C named Pens-plus        *
*  and simplified MPC structure which belongs to BBA together. MPC has been *
*  embeded into Pens-plus to enhance the performance                        *
*                                                        ————KuyuSi         *
*****************************************************************************
'''
"""
Created on Tue Dec 11 12:31:00 2018

@author: Administrator
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
#import pickle

SMOOTH_PENALTY= 0.02
REBUF_PENALTY = 1.5
LANTENCY_PENALTY = 0.005

GRADIENT_BATCH_SIZE = 8
BIT_RATE = [500.0,1200.0]

M_IN_K = 1000.0
class Algorithm:
     def __init__(self):
     # fill your init vars
         self.buffer_size = [0.3] * 10
         #self.TP_buf = [0] * 5
         self.TP_buf = [0.15] * 50
         self.buffer_list = []
         self.index = 3
         self.bit_rate = 1
         self.target_buffer = 1.0
         self.reward = 0
         self.changenumber = 0
         self.delay = 0
         self.freezetime = 0
     # Intial 
     def Initial(self):
         self.last_bit_rate = 0
         self.last_target_buffer = 0
         self.switch = 0
     
     def Compute_QOE(self, cdn_flag, frame_time_len, bit_rate,rebuf_time,S_end_delay):
        '''QOE_reward'''
        self.bit_rate = bit_rate
        if not cdn_flag:
            reward_frame = frame_time_len * float(BIT_RATE[self.bit_rate]) / 1000  - REBUF_PENALTY * rebuf_time - LANTENCY_PENALTY * S_end_delay[-1]
        else:
            reward_frame = -(REBUF_PENALTY * rebuf_time)
        if self.bit_rate != self.last_bit_rate:
            self.switch = 1
            self.changenumber += 1 
            self.freezetime += rebuf_time
        reward_frame += -(self.switch) * SMOOTH_PENALTY * (1200 - 500) / 1000
     #Define your al
        return reward_frame
     def run(self, S_time_interval, S_send_data_size, S_frame_time_len, S_frame_type, S_real_quality, S_buffer_size, S_end_delay, rebuf_time, cdn_has_frame,cdn_flag, buffer_flag):
         # record your params
         self.buffer_size.append(S_buffer_size)
         self.buffer_size.pop(0)
         # your decision
         bit_rate = 1
         target_buffer_list = [0.3,0.5,0.7,1.0,1.2,1.5,2.0,2.5,3.0,3.5]
         target_buffer = target_buffer_list[self.index]
         cbuffer = [target_buffer*1.6/3,target_buffer + 0.1]
         if S_time_interval[-1] == 0: 
             TP = 0
         else:
             TP = S_send_data_size[-1]/S_time_interval[-1] / M_IN_K / M_IN_K
         if not TP == 0:
             self.TP_buf.append(TP)
             self.TP_buf.pop(0)
             
         if TP == 0: TP += 0
             
         if len(self.buffer_size[-1]) < 2:
             last_buffer_size = 0
         else :
             last_buffer_size = self.buffer_size[-1][-2]

         #linear function:
#         S_buffer_size[-1] = last_buffer_size + data_size_rk/TP_avg - 0.5
         TP_avg = np.mean(self.TP_buf)
#         TP_avg = 10 / ( 1 / self.TP_buf[-1] + 1 / self.TP_buf[-2] +1 / self.TP_buf[-3] +1 / self.TP_buf[-4] +\
#                        1 / self.TP_buf[-5] +1 / self.TP_buf[-6] +1 / self.TP_buf[-7] + 1 / self.TP_buf[-8] + \
#                        1 / self.TP_buf[-9] + 1 / self.TP_buf[-10])
         data_size_rk = (S_buffer_size[-1] - last_buffer_size + 0.5) * TP_avg
         if self.last_bit_rate == 0:
             data_size_0 = data_size_rk
             data_size_1 = data_size_rk / 5 * 12
         elif self.last_bit_rate == 1:
             data_size_0 = data_size_rk / 12 * 5
             data_size_1 = data_size_rk
         
         time_0 = data_size_0/TP
         time_1 = data_size_1/TP
         fluct = (np.max(self.TP_buf) - np.min(self.TP_buf))/10
         
#         reward_list = []
#         target_buffer = 0.3
#         self.target_buffer = target_buffer
##             target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) + 1]
         '''qiangwang'''
#         if (S_buffer_size[-1] >= time_1 or S_buffer_size[-1] >= cbuffer[1]) and TP_avg <= self.TP_buf[-1]:
#             bit_rate = 1
#         elif S_buffer_size[-1] < time_0 and (S_buffer_size[-1] <= cbuffer[0]) or TP_avg > self.TP_buf[-1]:
#             bit_rate = 0
         '''ruowang/origin'''
         if (S_buffer_size[-1] >= time_1 or S_buffer_size[-1] >= cbuffer[1]) and np.max(self.TP_buf) - fluct <= self.TP_buf[-1]:
             bit_rate = 1
             target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) - 1]

         elif S_buffer_size[-1] < time_0 and S_buffer_size[-1] <= cbuffer[0] or np.min(self.TP_buf) + fluct > self.TP_buf[-1]:
             bit_rate = 0
             target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) + 1]

         
         if abs(self.TP_buf[-1] - self.TP_buf[-2]) >= TP_avg or abs(self.TP_buf[-2] - self.TP_buf[-3])  >= TP_avg or self.TP_buf[-2] + self.TP_buf[-1] <= TP_avg:# or recent < TP_avg * 4:
             bit_rate = 0
#             target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) + 1]
         recent = self.TP_buf[-2] + self.TP_buf[-1] + self.TP_buf[-3]# + self.TP_buf[-4]# + self.TP_buf[-5] + self.TP_buf[-6]
         minmax = (self.TP_buf[-2] - self.TP_buf[-1]) + (self.TP_buf[-3] - self.TP_buf[-2])
 
         '''ruowang''' 
         if TP_avg <= 0.87:
             bit_rate = 1
             if S_buffer_size[-1] >= cbuffer[1]:
                 bit_rate = 1
                 target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) - 1]
             elif S_buffer_size[-1] <= cbuffer[0]:
                 bit_rate = 0
            
             if recent < TP_avg * 3 or abs(self.TP_buf[-1] - self.TP_buf[-2]) >= TP_avg: 
                 bit_rate = 0
                 target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) + 1]
         '''qiangwang'''
         if TP_avg >= 1.00:
             bit_rate = 1
             if S_buffer_size[-1] >= cbuffer[1]:
                 bit_rate = 1
                 target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) - 1]
             elif S_buffer_size[-1] <= cbuffer[0]:
                 bit_rate = 0
                 target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) + 1]

             if recent >= TP_avg * 3 or abs(self.TP_buf[-1] - self.TP_buf[-2]) <= minmax/2:
                 bit_rate = 1
#                 target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) - 1]
         '''zhongwang'''
         if TP_avg < 1.02 and TP_avg > 0.85:
             bit_rate = 1
             cbuffer = [target_buffer*1.8/3,target_buffer + 0.1]
             if S_buffer_size[-1] >= cbuffer[1]:
                 bit_rate = 1
                 target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) - 1]
             elif S_buffer_size[-1] <= cbuffer[0]:
                 bit_rate = 0
                 target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) + 1]

             if recent >= TP_avg * 3 and abs(self.TP_buf[-1] - self.TP_buf[-2]) <= minmax/2:
                 bit_rate = 1
#                 target_buffer = target_buffer_list[target_buffer_list.index(target_buffer) - 1]
         reward = self.Compute_QOE(cdn_flag, S_frame_time_len[-1], bit_rate,rebuf_time,S_end_delay)
         self.last_bit_rate = self.bit_rate
         self.delay += S_end_delay[-1]
#         self.last_target_buffer = self.target_buffer S_buffer_size[-1]) / S_buffer_size[-1])))
         self.buffer_list.append([TP_avg])

         '''plot TP'''
#         x = np.arange(0,10)
#         y = self.TP_buf
##         print(y)
#         plt.plot(x,y)
#         plt.show()
         

         return bit_rate, target_buffer