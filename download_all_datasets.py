import moxing as mox
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

local_base = Path('/cache/wwx1484778/Pretraindata')
obs_base = 'obs://yw-2030-gy/external/wwx1484778/Pretraindata'

# 数据集列表
datasets = [
    'GM-100',
    'RC1.0',
    'RC2.0',
    'RDT',
    'RoboCoin',
    'RoboMIND',
    'RoboMIND2.0',
    'RoboTwin2.0',
]

lock = threading.Lock()

def download(name):
    src = f'{obs_base}/{name}'
    dst = local_base / name
    try:
        mox.file.copy_parallel(src, str(dst))
        with lock:
            print(f'✓ {name} 下载完成')
    except Exception as e:
        with lock:
            print(f'✗ {name} 失败: {e}')

print(f'开始下载 {len(datasets)} 个数据集...')
print()

with ThreadPoolExecutor(max_workers=8) as executor:
    executor.map(download, datasets)

print()
print('全部完成')