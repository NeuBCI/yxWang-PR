# Document of the Project

## The Dataset

- 数据来源：DREMAER数据集中某源被试和某目标被试的dominance轴的数据
- 数据内容：来自源被试的训练数据train_data，大小为708×14×14；来自目标被试的训练数据transfer_data，大小为236×14×14。来自目标被试的测试数据test_data，大小为295×14×14。其中14×14指的是在14个导联上提取的1s内时间序列信号的协方差矩阵。训练标签和测试标签均有2种类别，用0,1标识，代表dominance轴上的高和低两种状态。

## The Analysis

- 本算法主要部分包括对称正定矩阵网络，原型学习，对抗学习三个部分
  - 对称正定矩阵网络：SPDnet.py
  - 原型学习：function.py
  - 对抗学习：main.py（包含在主文件中）
- 数据输入部分：dataset.py
- 算法主文件：main.py

## Expected Output

- 最终输出为
  The last acc is : XXXXXX
  The peak acc is : XXXXXX