# 特征提取
import os
import numpy as np
import pandas as pd
# from pyentrp import entropy as ent

# OC0~16手动修改
denoised_signal_folder = 'Denoised\WT-EMD\OC16'
output_folder = 'Extracted'

# 主频率
def dominant_frequency(signal, sampling_rate=1250):
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate) # 计算频率轴
    magnitude = np.abs(fft_result) # 获取频谱幅值
    dominant_freq_index = np.argmax(magnitude[:len(magnitude) // 2]) # 找到主频率的索引
    dominant_freq = np.abs(freqs[dominant_freq_index])
    return dominant_freq

# 中心频率
def central_frequency(signal, sampling_rate=1250):
    fft_result = np.fft.fft(signal) # 对信号进行傅里叶变换
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate) # 计算频率轴
    magnitude = np.abs(fft_result) # 获取频谱幅值
    # 计算加权平均频率
    central_freq = np.sum(freqs[:len(freqs) // 2] * magnitude[:len(magnitude) // 2]) / np.sum(magnitude[:len(magnitude) // 2])
    return central_freq

""" # 频带宽度
def bandwidth(signal, sampling_rate=1250):
    # 对信号进行傅里叶变换
    fft_result = np.fft.fft(signal)
    # 计算频率轴
    freqs = np.fft.fftfreq(len(signal), 1 / sampling_rate)
    # 获取频谱幅值
    magnitude = np.abs(fft_result)
    # 计算功率谱密度
    psd = magnitude ** 2
    # 计算信号的平均功率
    avg_power = np.mean(psd)
    # 找到大于平均功率的频率范围
    indices = np.where(psd > avg_power)[0]
    bandwidth = freqs[indices[-1]] - freqs[indices[0]]
    return bandwidth """

# 峰度
def kurtosis(signal):
    mean_signal = np.mean(signal) # 计算信号的均值
    fourth_moment = np.mean((signal - mean_signal) ** 4) # 计算信号的四阶矩（中心化后）
    second_moment = np.mean((signal - mean_signal) ** 2) # 计算信号的二阶矩（中心化后）
    kurtosis = fourth_moment / (second_moment ** 2) # 峰度公式
    return kurtosis

# 均方根
def RMS(signal):
    rms_value = np.sqrt(np.mean(signal ** 2))
    return rms_value

# 峰值因子
def crest_factor(signal):
    max_value = np.max(np.abs(signal)) # 计算信号的最大值和均方根
    rms_value = RMS(signal)
    crest_factor = max_value / rms_value # 计算峰值因子
    return crest_factor

""" # 香农熵
def shannon_ent(signal_data):
    return ent.shannon_entropy(signal_data) """

""" # 样本熵
def sample_ent(signal_data, m=2, r=0.2):
    return ent.sample_entropy(signal_data, m, r) """

# 特征提取
def feature_extract(denoised_signal_folder, output_folder):
    feature_list=[]
    folder_name = os.path.basename(denoised_signal_folder)
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(denoised_signal_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(denoised_signal_folder, filename)
            df = pd.read_csv(file_path)
            voltage = df['电压(V)'].values
            # 计算特征
            dominant_freq = dominant_frequency(voltage)     # 主频率
            central_freq = central_frequency(voltage)       # 中心频率
            kurto = kurtosis(voltage)                       # 峰度
            rms = RMS(voltage)                              # 均方根
            crest = crest_factor(voltage)                   # 峰值因子
            # 存储特征
            feature_list.append([folder_name, 
                                 dominant_freq, 
                                 central_freq, 
                                 kurto, 
                                 rms, 
                                 crest
                                 ])
    # 将特征保存为DataFrame并输出为CSV文件
    features_df = pd.DataFrame(feature_list, columns=['label', 
                                                      'dominant frequency', 
                                                      'central frequency', 
                                                      'kurtosis', 
                                                      'RMS', 
                                                      'crest factor'
                                                      ])
    output_file = os.path.join(output_folder, f'{folder_name}.csv')
    features_df.to_csv(output_file, index=False)

feature_extract(denoised_signal_folder, output_folder)
