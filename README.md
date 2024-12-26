# 模式识别结课课程设计
## 说明
## 只有完整代码
考虑到数据量大小只使用各个工况70%的数据
将使用的数据储存在'Data'文件夹目录下
## WT-EMD去噪
已经处理过不需要再次运行
运行WT-EMD.py文件以进行WT-EMD去噪
如果需要运行请注意需要手动设置工况
'Denoised\WT-EMD'文件夹目录下的WT-EMD去噪信号即为最终用于特征提取的信号数据
## 特征提取
运行Feature Extraction.py文件以进行特征提取
注意需要手动选择需要的工况\n
被选中的工况将会在下一步被合并为训练集Train.csv
## 特征值均值图像绘制与训练集生成
运行Graph & Merge.py文件以进行此步骤
生成的训练集Train.csv保存在'Train'目录下
生成的特征值均值图像保存在'Graphs'目录下
## 模型选择
### 支持向量机
运行SVM.py文件
混淆矩阵图片保存在'Graphs'目录下
### 随机森林
运行Random Forest.py文件
混淆矩阵图片保存在'Graphs'目录下
## 依赖项
numpy 1.26.4
pandas 2.1.1
pyetnrp 1.0.0
matplotlib 3.8.0
scikit-learn 1.3.1
EMD-signal 1.6.4
PyWavelets 1.8.0
seaborn 0.12.2
