# WT-EMD去噪
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PyEMD import EMD
import pywt

# 文件目录
original_directory = 'Data\OC0' # 原始信号目录
emd_directory = 'Denoised\WT-EMD\OC0' # EMD去噪信号目录

# 读取CSV文件
def load_csv(file_path):
    # 读取电压(V)和x轴(s)数据
    data = pd.read_csv(file_path, encoding='gbk') # 以GBK编码读取原始文件否则表头乱码无法读取
    voltage = data['电压(V)'].values
    time = data['x轴(s)'].values
    return time, voltage

# 小波去噪函数
def wavelet_denoising(signal, wavelet='haar', level=4, threshold=0.05):
    # 小波变换
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # 阈值处理
    coeffs_thresholded = [pywt.threshold(c, threshold * np.max(c), mode='soft') for c in coeffs]
    # 小波重构
    denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)
    
    return denoised_signal

# 基于标准差和能量自适应计算去噪阈值
def adaptive_threshold(imf, k_std=0.2, energy_ratio_threshold=0.05):
    """
    根据IMF的标准差和能量计算自适应阈值
    k_std: 标准差的倍数阈值
    energy_ratio_threshold: 能量比阈值 用于判断是否为噪声IMF
    """
    std_dev = np.std(imf) # 计算IMF的标准差
    energy = np.sum(imf ** 2) # 计算IMF的能量 使用方差来近似能量
    total_energy = np.sum(imf ** 2) # 计算该IMF的能量与所有IMF的能量的比值
    threshold_std = k_std * std_dev # 计算IMF的标准差阈值
    # 判断是否去除IMF
    # 如果IMF的能量比非常小且标准差小于阈值，则认为是噪声
    if energy / total_energy < energy_ratio_threshold and std_dev < threshold_std:
        return True # 被认为是噪声
    else:
        return False # 被认为是信号

# EMD去噪函数
def emd_denoising(signal, k_std=0.2, energy_ratio_threshold=0.05):
    """
    EMD去噪函数 使用自适应阈值去除噪声IMF
    k_std: 标准差倍数 决定去噪的强度
    energy_ratio_threshold: 能量阈值 判断IMF是否为噪声
    """
    emd = EMD() # 执行经验模态分解
    imfs = emd.emd(signal)
    # 对每个IMF使用自适应阈值判断是否去噪
    denoised_imfs = []
    for imf in imfs:
        if adaptive_threshold(imf, k_std, energy_ratio_threshold):
            denoised_imfs.append(np.zeros_like(imf)) # 如果是噪声IMF，设置为零
        else:
            denoised_imfs.append(imf) # 保留信号IMF
    denoised_signal = np.sum(denoised_imfs, axis=0) # 重构去噪后的信号
    
    return denoised_signal

# 计算信噪比(SNR)(dB)
def calculate_snr(original_signal, denoised_signal):
    noise = original_signal - denoised_signal
    signal_power = np.sum(original_signal**2)
    noise_power = np.sum(noise**2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# 计算平均绝对误差(MAE)
def calculate_mae(original_signal, denoised_signal):
    return mean_absolute_error(original_signal, denoised_signal)

# 计算均方误差(MSE)
def calculate_mse(original_signal, denoised_signal):
    return mean_squared_error(original_signal, denoised_signal)

# 保存去噪后的信号
def save_denoised_signal(file_path, time, denoised_signal, output_dir, suffix):
    # 创建目标文件路径
    file_name = os.path.basename(file_path)
    name, ext = os.path.splitext(file_name)
    output_file = os.path.join(output_dir, name + suffix + ext)
    # 创建一个DataFrame保存去噪后的信号
    output_data = pd.DataFrame({'电压(V)': denoised_signal, 'x轴(s)': time})
    # 保存为CSV文件
    output_data.to_csv(output_file, index=False)

# 保存图像
def save_plot(time, original_signal, denoised_signal, snr, mae, mse, file_name, output_dir):
    # 创建图像文件名
    plot_file = os.path.join(output_dir, f'{file_name}_WT-EMD.png')
    plt.figure(figsize=(8, 5))
    plt.plot(time, original_signal, label='Original Signal')
    plt.plot(time[:len(denoised_signal)], denoised_signal, label='EMD Denoised Signal')
    plt.title(f'EMD Denoising (SNR: {snr:.2f} dB, MAE: {mae:.4f}, MSE: {mse:.4f})')
    plt.xlabel('Time(s)')
    plt.ylabel('Voltage(V)')
    plt.legend()
    plt.savefig(plot_file)
    # plt.show()
    plt.close()

# 主流程：加载数据 -> 小波去噪 -> EMD去噪 -> 可视化结果
def process_signal(file_path, original_dir, emd_dir):
    # 1.加载原始数据
    time, original_signal = load_csv(file_path)
    # 2.小波去噪
    wt_signal = wavelet_denoising(original_signal)
    # 3.将WT去噪后的信号输入到EMD去噪
    emd_signal = emd_denoising(wt_signal)
    # 4.计算SNR（与原始信号对比）
    snr = calculate_snr(original_signal, emd_signal)
    # 5.计算MAE（与原始信号对比）
    mae = calculate_mae(original_signal, emd_signal)
    # 6.计算MSE（与原始信号对比）
    mse = calculate_mse(original_signal, emd_signal)
    # 保存EMD去噪后的信号
    save_denoised_signal(file_path, time, emd_signal, emd_dir, suffix='_WT-EMD')
    
    # 保存图像
    save_plot(time, original_signal, emd_signal, snr, mae, mse, os.path.basename(file_path), emd_dir)
    
    return emd_signal, snr, mae, mse

# 处理Data目录下的所有信号
def process_all(original_directory, emd_directory):
    files = [f for f in os.listdir(original_directory) if f.endswith('.csv')]
    os.makedirs(emd_directory, exist_ok=True) # 不存在则创建目录
    
    for file in files:
        original_file_path = os.path.join(original_directory, file)
        print(f'Processing file: {original_file_path}')

        emd_signal, snr, mae, mse = process_signal(original_file_path, original_directory, emd_directory)
        
        print(f'File: {file}')
        print(f'Signal-to-Noise Ratio (SNR): {snr:.2f} dB')
        print(f'Mean Absolute Error (MAE): {mae:.4f}')
        print(f'Mean Squared Error (MSE): {mse:.4f}')
        print('-' * 50)

process_all(original_directory, emd_directory)
