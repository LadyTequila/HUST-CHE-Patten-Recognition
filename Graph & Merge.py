# 特征提取效果图
# 合并CSV文件
import os
import pandas as pd
import matplotlib.pyplot as plt

directory_path = 'Graphs'
input_folder = 'Extracted'
output_file = 'Train\Train.csv'

def merge(input_folder, output_file):
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    merged_df = pd.DataFrame()
    
    for file in files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    merged_df.to_csv(output_file, index=False)

file_means = []
for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        # 生成完整文件路径
        file_path = os.path.join(input_folder, filename)
        
        df = pd.read_csv(file_path)
        
        # 检查文件是否有足够的列
        if df.shape[1] >= 6:
            # 计算每列的均值
            dominant_frequency_mean = df.iloc[:, 1].mean()
            central_frequency_mean = df.iloc[:, 2].mean()
            # bandwidth_mean = df.iloc[:, 3].mean()
            kurtosis_mean = df.iloc[:, 3].mean()
            rms_mean = df.iloc[:, 4].mean()
            crest_factor_mean = df.iloc[:, 5].mean()
            # shannon_entropy_mean = df.iloc[:, 4].mean()
            
            file_means.append((filename, 
                               dominant_frequency_mean, 
                               central_frequency_mean, 
                               # bandwidth_mean,
                               kurtosis_mean, 
                               rms_mean, 
                               crest_factor_mean, 
                               # shannon_entropy_mean
                               ))

mean_df = pd.DataFrame(file_means, columns=['Filename', 
                                            'Dominant Frequency', 
                                            'Central Frequency', 
                                            'Kurtosis', 
                                            'RMS', 
                                            'Crest Factor', 
                                            ])

# 绘制图表
plt.figure(figsize=(8, 5))
# dominant frequency
plt.plot(mean_df['Filename'], 
         mean_df['Dominant Frequency'], 
         marker='o', 
         label='Dominant Frequency', 
         linestyle='-', 
         color='skyblue')
# central frequency
plt.plot(mean_df['Filename'], 
         mean_df['Central Frequency'], 
         marker='o', 
         label='Central Frequency', 
         linestyle='-', 
         color='orange')
# kurtosis
plt.plot(mean_df['Filename'], 
         mean_df['Kurtosis'], 
         marker='o', 
         label='Kurtosis', 
         linestyle='-', 
         color='blue')
# rms
plt.plot(mean_df['Filename'], 
         mean_df['RMS'], 
         marker='o', 
         label='RMS', 
         linestyle='-', 
         color='pink')
# crest factor
plt.plot(mean_df['Filename'], 
         mean_df['Crest Factor'], 
         marker='o', 
         label='Crest Factor', 
         linestyle='-', 
         color='purple')

plot_file = os.path.join(directory_path, f'Mean Values.png')
plt.xlabel('Filename')
plt.ylabel('Mean Value')
plt.title('Mean Values')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.savefig(plot_file)
plt.show()

merge(input_folder, output_file)
