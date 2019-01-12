# Pensieve-A3C-Streaming-Adaptive-Bitrate-Model
based on Pensieve model to figure ABR problem in general streaming

## 环境安装

--pip install LiveStreamingEnv==0.7.1

## 仿真器SIM

功能:类似于一个播放器系统，模拟播放器在不同网络环境下下载视频帧、并动态播放的过程。每次调度为下载了一帧。

仿真器输入，输入分为三部分:

      1.Frame Trace,视频数据集，此数据集重在模拟视频源的动态变化情况。详细请看数据集

      2.Network Trace,网络数据集，此数据集重在模拟网络的动态变化情况。详细请看数据集

      3.ABR算法的决策，即码率和目标缓冲区大小

仿真器输出:仿真器的各个指标的状态。所含指标有：物理时间、当前下载帧、当前播放时间、客户端缓冲区大小、当前网络RTT等。详细请看下表：

1、time（s）	物理时间
2、time_interval（s）	本周期经过的物理时间
3、send_data_size:（bit）	本周期下载的数据量
4、chunk_len(s)	当前要下载帧的时间长度
5、rebuf(s)	本周期内的卡顿时间
6、buffer_size(s)	当前时刻的缓冲区大小
7、RTT（ms）	当前网络的RTT
8、play_time_len(s)	本周期播放的时间长度
9、end_delay(s)	当前端到端时延
10、decision_flag(Flase/True)	是否到GOP边界
11、buffer_flag(Flase/True)	播放器是否在缓冲
12、cdn_rebuffer_flag(Flase/True)	CDN是否有可取的帧
13、end_of_video(Flase/True)	视频结束标志
14、download_id	当前下载帧的id
15、cdn_has_frame	CDN帧集合
16、cdn_newest_id	CDN上最新一帧的id
表1 仿真器返回值

运行环境:Python2 or Python 3 Linux/Mac or Windows

安装:我们已经在python的官方源中将仿真器封装好。大家只需要执行以下命令：

pip install LiveStreamingEnv==0.5.2


C. ABR Algorithm
ABR Algorithm根据仿真器提供的状态信息，做出码率和目标缓冲区大小决策。为了方便大家更好地上手，我们提供了一个快速开始SDK的DEMO

本Demo提供了根据客户端buffer去做调度的一个样例，选手只需要在填充demo的算法部分。在demo中我们也提供了动态画图的功能，只需要将DRAW设置为True。就可以看到实时画图的结果。

具体文档详细请看git上的README

下载地址: git clone https://github.com/NGnetLab/LiveStreamingDemo


D. 用户体验质量（QoE)
QoE与以下指标有关

指标	变量名	权重（正/负）（奖励/惩罚）
码率	Bitrate	+
卡顿时间	Rebuff	-
时延	Latency	-
切换频次	Smooth	-
表2 评测指标

    QOE分两部分：一部分为frame QOE，一部分为Gop QOE

QOE = QOE1 + QOE2

     1、对于每一帧，公式为：

QoE1 = play_time_duration（播放时长） * Bitrate - 1.5 * rebuff- 0.005 * latency

    2、对于每一块，公式为：

QOE2 = - 0.02 *smooth

    详细请看DEMO 文件

