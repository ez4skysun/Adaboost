utils.py：获取数据 处理数据的函数 其中包括kmeans对每日数据进行聚类从而对收益率取均值降噪
	get_data 取数据
	process_data 处理数据
adaboost.py：adaboost模型的主要框架
	参数：max_depth min_sample_split是弱分类器决策树的参数，分别为最大深度与叶子节点最小划分比
		learning_rate k n_estimators是adaboost框架参数，分别为学习率，TP偏好系数，最大弱分类器个数
		early_stop是早停法参数，True启动早停法，40个弱分类器后若连续25个弱分类器加入强分类器时在验证集的准确度没有超过峰值则回退到峰值模型
train.py：三种模型的训练函数 这一part比较简单
		主函数里把所有训练过程都写好了

两个ipynb文件：
adaboost.ipynb：是基于短期数据去做基础参数的调试
滚动测试.ipynb：是基于长期数据去做滚动训练的调试

20_3_factor
60_daily_factor
60_monthly_factor分别是三种模型训练得出的因子数据
但是没有做归一化处理，而且删除了stockcode列。但是新的主函数计算的因子是有stockcode的

FactorData.h5是滚动测试的数据集，2017年至2022年4月的因子数据以及收益率数据

具体各个函数都有在代码部分做了简单注释