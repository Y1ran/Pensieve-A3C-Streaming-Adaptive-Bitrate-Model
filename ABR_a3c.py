import a3c
import tensorflow as tf
import numpy as np

'''pre-setting of paramaters'''
S_INFO = 5
S_LEN = 24
A_DIM = 2
T_DIM = 2
ACTOR_LR_RATE = 1e-4
CRITIC_LR_RATE = 1e-3
SMOOTH_PENALTY= 0.02
REBUF_PENALTY = 1.5
LANTENCY_PENALTY = 0.005

GRADIENT_BATCH_SIZE = 8
BIT_RATE = [500.0,1200.0]
#TARGET_BUFFER = [0.3,1.0,1.5,2.0,2.5,3.0,3.5,4.0]
TARGET_BUFFER = [0.3,0.5,1.0,1.5]
#[0.5,1.25,2.25]   91.75006604104237
#[0.5,1.0,2.0]  92.24513363213242
#[0.5,1.0,2.0,3.0] 83.71735234080244

#self.target_buffer = 1   
FPS = 25

DEFAULT_QUALITY = 1
TRAIN_SEQ_LEN = 100
RAND_RANGE = 1000.0
frame_time_len = 0.04
reward_all_sum = 0
M_IN_K = 1000.0
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 68.0

LogFile_Path = "./log/"

class Algorithm:
     def __init__(self):
     # fill your init vars
        self.buffer_size = [0] * 20
        
        self.offline = False
        self.epoch = 0
        self.last_bit_rate = 1
        self.bit_rate = 1
        self.target_buffer = 1
        self.last_target_buffer = 1
     # QOE setting
        self.action_vec = np.zeros(A_DIM)
        self.a_batch = [self.action_vec]
        self.s_batch = [np.zeros((S_INFO, S_LEN))]
        self.r_batch = []
        self.entropy_record = []
        self.reward_frame = 0
        self.reward_all = 0
        self.actor_gradient_batch = []
        self.critic_gradient_batch = []
        self.reward_frame_old = 0
        self.switch = 0
        self.last_target_buffer = 0
     # Intial 
     def Initial(self):
         
        with tf.Session().as_default() as sess:
            saver = tf.train.import_meta_graph('log/nn_model_ep_60.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint("log/"))
            print("Model restored.")
            actor = a3c.ActorNetwork(sess,
                                     state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                     learning_rate=ACTOR_LR_RATE)
    
            critic = a3c.CriticNetwork(sess,
                                     state_dim=[S_INFO, S_LEN],
                                     learning_rate=CRITIC_LR_RATE)
            print("init successe \n")
            sess.run(tf.global_variables_initializer())
    #             saver = tf.train.Saver()  # save neural net parameters
            print("saver created")
                # restore neural net parameters
    #             if NN_MODEL is not None:  # NN_MODEL is the path to file
    #                 saver.restore(sess, NN_MODEL)
    #                 print("Testing model restored.")
    #             print("Nnmodel restored")
            self.actor = actor
            self.critic = critic
            self.sess = sess
            self.TP_buf = [0.25] * 125

     #Define your al
     def run(self, S_time_interval, S_send_data_size, S_frame_time_len, S_frame_type, S_real_quality, S_buffer_size, S_end_delay, rebuf_time, cdn_has_frame,cdn_flag, buffer_flag):
         # record your params
         self.buffer_size.append(S_buffer_size)
         self.buffer_size.pop(0)
#         
         
         actor = self.actor
         critic = self.critic


         # retrieve previous state
         if len(self.s_batch) == 0:
             state = [np.zeros((S_INFO, S_LEN))]
         else:
             state = np.array(self.s_batch[-1], copy=True)
    
         # dequeue history record
         if S_time_interval[-1] == 0: TP = 0
         else:
             TP = S_send_data_size[-1]/S_time_interval[-1] / M_IN_K / M_IN_K


         self.last_tp = TP
         avg = np.average(self.TP_buf)
         
         if len(self.buffer_size[-1]) < 2:
             self.last_target_buffer = 0
         else:
             self.last_target_buffer = self.buffer_size[-1][-2]
             
         state = np.roll(state, -1, axis=1)
         # video_chunk_remain = S_buffer_size[-1] - S_play_time_len[-1]
         # this should be S_INFO number of terms


         state[0, -1] = TP / BUFFER_NORM_FACTOR
         state[1, -1] = S_buffer_size[-1] / BUFFER_NORM_FACTOR  # 10 sec
         state[2, -1] = rebuf_time / BUFFER_NORM_FACTOR
         state[3, -1] = S_end_delay[-1] / BUFFER_NORM_FACTOR
         state[4, -1] = BIT_RATE[self.bit_rate] / float(np.max(BIT_RATE))
#         state[13, -1] = len(cdn_has_frame) * 0.04 + S_buffer_size[-1]
#         ckpt = tf.train.get_checkpoint_state('./log/')
         
#         if self.offline == True:
#             with tf.Session() as sess:
             
         action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
         action_cumsum = np.cumsum(action_prob)
    
                           
#         else:
#             action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))
#             action_cumsum = np.cumsum(action_prob)
                 
         self.target_buffer = (action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
             # Note: we need to discretize the probability into 1/RAND_RANGE steps,
             # because there is an intrinsic discrepancy in passing single state and batch states
                 
#         self.entropy_record.append(a3c.compute_entropy(action_prob[0]))

#         bit_rate = 1
#         target_buffer = 1.0
         cbuffer = [self.target_buffer*2/3, self.target_buffer + 0.1]
         if S_time_interval[-1] == 0: 
             TP = 0
         else:
             TP = S_send_data_size[-1]/S_time_interval[-1] / M_IN_K / M_IN_K
         if not TP == 0:
             self.TP_buf.append(TP)
             self.TP_buf.pop(0)
         #print("cdn_has_frame",cdn_has_frame)
         #print("S_real_quality",S_real_quality)
         #print("TP",TP)
         if avg < self.TP_buf[-1] or S_buffer_size[-1] >= cbuffer[1]:
             self.bit_rate = 1
         elif S_buffer_size[-1] <= cbuffer[0]:
             self.bit_rate = 0
             
#         if download_id == 49 or download_id == 99:
#             self.bit_rate = 0
         if self.offline == False:
             '''QOE_reward'''
             if not cdn_flag:
                reward_frame = frame_time_len * float(BIT_RATE[self.bit_rate]) / 1000  - REBUF_PENALTY * rebuf_time - LANTENCY_PENALTY * S_end_delay[-1]
             else:
                reward_frame = -(REBUF_PENALTY * rebuf_time)
             if self.bit_rate != self.last_bit_rate:
                self.switch = 1
                
             reward_frame += -(self.switch) * SMOOTH_PENALTY * (1200 - 500) / 1000
             
             self.last_bit_rate = self.bit_rate
             self.last_target_buffer = self.target_buffer
#             
#             self.reward_frame_old = reward_frame
#             self.r_batch.append(reward_frame)
##             else:
##                 self.r_batch.append(reward_frame2)
##                 self.bit_rate = bit2
#    
#                 
#             if len(self.r_batch) >= TRAIN_SEQ_LEN:  # do training once
#                 actor_gradient, critic_gradient, td_batch = \
#                            a3c.compute_gradients(s_batch=np.stack(self.s_batch[1:], axis=0),  # ignore the first chuck
#                                                  a_batch=np.vstack(self.a_batch[1:]),  # since we don't have the
#                                                  r_batch=np.vstack(self.r_batch[1:]),  # control over it
#                                                  terminal=False, actor=actor, critic=critic)
#                 td_loss = np.mean(td_batch)
#        
#                 self.actor_gradient_batch.append(actor_gradient)
#                 self.critic_gradient_batch.append(critic_gradient)
#        
#    #             print("================")
#    #             print("Epoch", self.epoch)
#    #             print ("TD_loss", td_loss, "Avg_entropy", np.mean(self.entropy_record), 
#    #                           '\n',
#    #                           "Avg_reward", np.mean(self.r_batch),"Total_reward", sum(self.r_batch))
#    #             print("================")
#        
#        
#                 if len(self.actor_gradient_batch) >= GRADIENT_BATCH_SIZE:
#        
#                     assert len(self.actor_gradient_batch) == len(self.critic_gradient_batch)
#                            # assembled_actor_gradient = actor_gradient_batch[0]
#                            # assembled_critic_gradient = critic_gradient_batch[0]
#    
#        
#                     for i in range(len(self.actor_gradient_batch)):
#                         actor.apply_gradients(self.actor_gradient_batch[i])
#                         critic.apply_gradients(self.critic_gradient_batch[i])
#        
#                     self.actor_gradient_batch = []
#                     self.critic_gradient_batch = []
#        
#                     self.epoch += 1
#                     
#    
##                     if self.epoch % 10 == 0:
##                         # Save the neural net parameters to disk.
##                         save_path = self.saver.save(self.sess, LogFile_Path + "/nn_model_ep_" +
##                                                      str(self.epoch) + ".ckpt")
##                         print("Model saved in file: %s" % save_path)
#    
#                 self.reward_all += np.sum(self.r_batch)
#                 
#                 del self.s_batch[:]
#                 del self.a_batch[:]
#                 del self.r_batch[:]
#
#         
#             self.s_batch.append(state)
#            
#             action_vec = np.zeros(A_DIM)
#             action_vec[self.target_buffer] = 1
#             self.a_batch.append(action_vec)
#            
         return self.bit_rate, self.target_buffer