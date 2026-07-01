# import moxing as mox
# dataset_path = 'obs://yw-ads-training-gy1/test.log' 
# local_path = '/home/ma-user/work/wwx1484778/claude_code/test.log' 
# mox.file.copy_parallel(dataset_path, local_path)

# import moxing as mox
# dataset_path = 'obs://yw-2030-gy/data/opensource/RoboTwin2_0' 
# local_path = '/home/ma-user/work/wwx1484778/claude_code/test.log' 
# mox.file.copy_parallel(local_path, dataset_path)


# import moxing as mox
# print('mox.file 可用方法:')
# methods = [m for m in dir(mox.file) if not m.startswith('_')]
# for m in sorted(methods):
#     print(f'  - {m}')

# # 测试每个可能相关的方法
# print('\n=== 测试方法 ===')
# test_dir = 'obs://yw-2030-gy/data/opensource/RoboTwin2_0/'

# candidates = ['list_directory', 'get_file_list', 'ls', 'dir', 'list_objects']
# for method in candidates:
#     if hasattr(mox.file, method):
#         try:
#             files = getattr(mox.file, method)(test_dir)
#             print(f'{method}: 成功, 找到 {len(files)} 个')
#         except Exception as e:
#             print(f'{method}: {e}')
#     else:
#         print(f'{method}: 方法不存在')


import moxing as mox

# 检查 OBS 上是否已有解压后的目录
obs_dir = 'obs://yw-2030-gy/data/opensource/RoboTwin2_0/dataset/adjust_bottle/aloha-agilex_clean_50/'

if mox.file.exists(obs_dir):
    print('目录存在!')
    files = mox.file.list_directory(obs_dir)
    print(f'包含 {len(files)} 项:')
    for f in files:
        print(f'  - {f}')
else:
    print('目录不存在，需要解压 ZIP')

