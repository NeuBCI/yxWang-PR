\

[Abstract]

---

## Published Paper

- Abstract

  Emotion plays a vital role in human's daily life and EEG signals are widely used in emotion recognition. Due to the individual variability, it's hard to train a generic emotion recognition model across different subjects. The conventional method is collecting a large number of calibration data to build subject-specific models. Recently, developing a Brain-Computer Interface (BCI) with a short calibration time has become a challenge. To solve this problem, we propose a domain adaptation SPD matrix network (daSPDnet), which can successfully catch an intrinsic emotional representation shared between different subjects. Our method exploits feature adaptation with distribution confusion and sample adaptation with centroid alignment jointly. We compute SPD matrix based on covariance as feature and have a novel attempt to combine the prototype learning with Riemannian metric. 

- Key Points

  EEG; emotion recognition; domain adaption; SPD matrix; Riemannian metric; prototype learning.

- Main Finding

  Extensive experiments are conducted on the DREAMER and DEAP datasets and results show the superiority of our proposed method.

## Technology

### Dataset and Output

我们使用了公开数据集DREAMER，下载地址：https://jbhi.embs.org/2017/12/18/dreamer-database-emotion-recognition-eeg-ecg-signals-wireless-low-cost-off-shelf-devices
以及公开数据集DEAP，下载地址：http://www.eecs.qmul.ac.uk/mmv/data-sets/deap/

主要输出如 PR.png 所示

#### Previous Knowledge

GAN; Rimannian metric; prototype learning;  

#### Proposed Idea

 Our work is the first attempt in the EEG emotion recognition to use the SPD matrix as feature. Comparing with Riemannian-based methods, our model takes full advantage of deep learning to solve domain adaption problem of SPD matrix.
 Our work novelly combines prototype learning with Riemannian metric. Using the prototype loss we propose, it's easy to calculate the geometric mean on the low-dimensional layer of the neural network.
 Our work transfers knowledge from two levels to match two probability distributions meanwhile. On feature level, we confuse the distributions from source and target domain. And on sample level, we adapt the marginal distributions with alignment of prototypes each category.


The key Codes in the Project and how to use the Codes.

cd /data/home/wyx/example
/data/software/anaconda2/bin/python /data/home/wyx/example/Code/PR/main.py
