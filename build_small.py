import PyInstaller.__main__
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

PyInstaller.__main__.run([
    'train_2048_0.1.py',

    # 打包成单个文件
    '--onefile',
    '--windowed',
    '--name=布里茨大战2048',

    '--exclude-module=matplotlib',
    '--exclude-module=scipy',
    '--exclude-module=pandas',
    '--exclude-module=torchvision',
    '--exclude-module=torchaudio',
    '--exclude-module=tensorboard',
    '--exclude-module=numba',
    '--exclude-module=cv2',
    '--exclude-module=PIL',
    '--exclude-module=pyautogui',

    # 排除不需要的 C 语言扩展
    '--exclude-module=_tkinter',
    '--exclude-module=tkinter',

    # 强制包含可能遗漏的隐式依赖
    '--hidden-import=torch',
    '--hidden-import=pygame',
    '--hidden-import=PyQt5',
    '--hidden-import=numpy',

    # 如果你有 UPX 压缩工具，可以加这一行进一步压缩（需先下载 upx.exe 放到路径）
    # '--upx-dir=D:\\tools\\upx',     # 替换成你的 upx 文件夹路径
])