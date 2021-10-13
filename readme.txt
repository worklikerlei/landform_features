pytorch训练环境要求如下:
1、pytorch版本号----torch1.5.0, torchvision__version0.6.0
2、python版本号----3.6
3、操作系统类型及版本号----Ubuntu16.04.5LTS;
4、cuda和cudnn的版本号----NVIDIA-SMI 440.82 Driver Version: 440.82 CUDA Version: 10.2;
5、gcc版本----5.4.0


稀疏特征部分的训练与测试：
python train_sparse.py
python test_sparse.py

浓缩特征部分的训练与测试：
python train_dense.py
python test_dense.py


整体测试：
1.分别保存稀疏部分和浓缩部分的检测框：
python save_sparse.py
python save_dense.py
2.整合测试
python test_add.py