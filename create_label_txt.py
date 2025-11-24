from os.path import join, exists, dirname, abspath
import numpy as np
import os, glob, pickle

dataset_path = '/root/autodl-tmp/Coral'
train_folder = join(dataset_path, 'Training')
test_folder = join(dataset_path, 'Validation')
out_folder = dirname(dataset_path)

# print( glob.glob(join(dataset_path, '*.txt') ))

for pc_path in glob.glob(join(test_folder, '*.txt')):
    print(pc_path)
    file_name = pc_path.split('/')[-1][:-4]

    # f = train_folder + filename + '.txt'
    x = np.loadtxt(pc_path, dtype=np.float,
                   delimiter=' ', usecols=(0, 1, 2), unpack=False)
    y = np.loadtxt(pc_path, dtype=np.float,
                   delimiter=' ', usecols=(3, 4, 5), unpack=False)
    w = np.loadtxt(pc_path, dtype=np.float,
                   delimiter=' ', usecols=(6), unpack=False)
    d = np.int32(w)
    print(len(x))
    z = np.ones(len(x))
    # z = np.int(z)
    # merge = np.concatenate((x,y[:,None],z[:,None],z[:,None],z[:,None]),axis=1)
    merge = np.concatenate((x, z[:, None], y), axis=1)
    np.savetxt(out_folder + '/' + file_name + '.txt', merge,
               fmt='%f', delimiter=' ')
    np.savetxt(out_folder + '/' + file_name + '.labels', d,
               fmt='%d', delimiter=' ')

# filename = '2023.11.08.2.1'
# f = '/media/wcq/移动硬盘/Code/KPConv-master/Data/Semantic3d/original_data/txt/' + filename + '.txt'
# x = np.loadtxt(f ,dtype=np.float,
#                    delimiter=' ',usecols=(0,1,2),unpack=False)
# y = np.loadtxt(f ,dtype=np.float,
#                    delimiter=' ',usecols=(3,4,5),unpack=False)
# w = np.loadtxt(f ,dtype=np.float,
#                    delimiter=' ',usecols=(6),unpack=False)
# d = np.int32(w)
# print(len(x))
# z = np.ones(len(x))
# # z = np.int(z)
# # merge = np.concatenate((x,y[:,None],z[:,None],z[:,None],z[:,None]),axis=1)
# merge = np.concatenate((x, z[:,None], y),axis=1)
# merge = np.concatenate((x, np.int32(z[:,None])),axis=1)
# z = np.zeros((len(x),4))
# merge = np.concatenate((x,z),axis=1)

# with open('/home/mh/Disk_rfdnet/wzy/randla-net-tf2-main/data/test/3.txt' , 'w') as f:
#     for row in merge:
#         row_str = [str(col) for col in row]
#         # 将每行的数据使用制表符连接成一行字符串
#         row_text = '\t'.join(row_str)
#         # 将每行的数据写入文件
#         f.write(row_text + '\n')

# np.savetxt(r'/media/wcq/移动硬盘/Code/KPConv-master/Data/Semantic3d/original_data/' + filename + '.txt', merge, fmt='%f', delimiter=' ')
# np.savetxt(r'/media/wcq/移动硬盘/Code/KPConv-master/Data/Semantic3d/original_data/' + filename + '.labels', d, fmt='%d', delimiter=' ')


# f = '/home/mh/Disk_randlanet/wzy/haigong_kpconv/Data/new/231s4.txt'
# x = np.loadtxt(f ,dtype=np.float,skiprows=1,
#                    delimiter=' ',usecols=(0,1,2),unpack=False)
# y = np.loadtxt(f ,dtype=np.float,skiprows=1,
#                    delimiter=' ',usecols=(3,4,5),unpack=False)
# w = np.loadtxt(f ,dtype=np.float,skiprows=1,
#                    delimiter=' ',usecols=(6),unpack=False)
# d = np.int32(w)
# print(len(x))
# z = np.zeros(len(x))
# merge = np.concatenate((x,z[:,None],y),axis=1)
# # z = np.zeros((len(x),4))
# # merge = np.concatenate((x,z),axis=1)
#
# np.savetxt(r'/home/mh/Disk_randlanet/wzy/haigong_kpconv/Data/Semantic3D/original_data/231s4.txt', merge, fmt='%f', delimiter=' ')
# np.savetxt(r'/home/mh/Disk_randlanet/wzy/haigong_kpconv/Data/Semantic3D/original_data/231s4.labels', d, fmt='%d', delimiter=' ')

