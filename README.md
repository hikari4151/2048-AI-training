# 2048-AI-training
基于python的DQN实现让ai深度学习训练2048小游戏/A Python-based DQN implementation enables AI to deep learn and train in the 2048 mini-game
程序自行部署打开默认自动运行，未来会制作exe的麻瓜版本

用到的python依赖
1：torch

2：numpy

3：pygame

4：pyqt5

关于打包exe程序
#第一步在pycharm中打开下方的终端
#第二步输入build_env\Scripts\activate进入虚拟环境
#第三步输入pip install pyinstaller torch numpy pygame pyqt5安装库
#第四步输入python build_small.py进行打包（因为依赖库内存很大所以会等上10分钟左右，耐心等待）
