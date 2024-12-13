import sys

import cv2
import numpy as np
from hashlib import sha256

from PIL.Image import Image
from matplotlib import pyplot as plt

file_path = 'result/feature_list.txt'
data_list = []

test_list = []

sum = 0
# 打开输入文件
with open(file_path, 'r') as file:
    # 逐行读取文件内容
    lines = file.readlines()
    for line in lines:
        # 检查当前行是否包含文件名
        if '.png' not in line:
            # 分离并转换每一行的数字字符串
            num_strs = line.split()
            data = [int(num_str) for num_str in num_strs]
            if data:
                if sum < 450:
                    data_list.append(data)
                else:
                    test_list.append(data)
                sum = sum + 1;

# 生成 LWE 秘钥
q = 12289  # 大质数
n = 2048  # 向量长度
m = 2048  # 矩阵行数

A = np.random.normal(size=(m, n))  # 随机分布矩阵
s = np.random.randint(0, q, size=n)  # 噪声向量
e = np.random.normal(scale=5.0, size=m)  # 随机噪声向量

A_shuffled = A[np.random.permutation(m)]  # 打乱矩阵行排列
SA = (np.dot(A_shuffled, s) + e) % q  # 计算 S·A + e (模 q)

sk = s.astype(np.uint16).tolist()  # 私有密钥，将 s 向量转换为 Python list 类型
pk = [A_shuffled.tolist(), SA.tolist()]  # 公开密钥，将 A 和 SA 转换为 Python list 类型


def lwe_encrypt_vector(v):
    """ 对向量 v 进行 LWE 加密 """
    encrypted_v = []
    for i in range(len(v)):
        r = np.random.randint(0, q)
        c = ((r * A[i]).sum() + np.random.normal(scale=2.5)) % q
        encrypted_v.append((c + v[i] + s[i] * r) % q)
    return encrypted_v


def lwe_generate_index(lst, num_buckets):
    """ 生成 LWE 哈希索引 """
    indexes = []
    bucket_size = len(lst) // num_buckets
    for i in range(num_buckets):
        bucket = lst[i]
        encrypted_bucket = lwe_encrypt_vector(bucket)
        hash_key = sha256(bytes(str(i), 'utf-8')).hexdigest()
        indexes.append((hash_key, encrypted_bucket))

    return indexes


def lwe_search_index(indexes, target_v):
    """在LWE索引中查找与目标向量相似的向量"""

    # Step 1: 加密目标向量
    encrypted_target_v = lwe_encrypt_vector(target_v)
    # print(encrypted_target_v)

    # Step 2: 查找最相似的加密向量
    max_similar = -np.inf  # 相似度初始值为负无穷
    similar_weight = 0.9  # 权重系数

    best_key = None
    for i, (hash_key, encrypted_v) in enumerate(indexes):

        # Step 3: 计算加密向量和目标向量之间的相似度（余弦相似度）
        encrypted_v = np.array(encrypted_v)  # 将Python list转换为NumPy array
        encrypted_target_v = np.array(encrypted_target_v)
        sim = np.dot(encrypted_v, encrypted_target_v) / (
                np.linalg.norm(encrypted_v) * np.linalg.norm(encrypted_target_v))
        # sim_score = sim
        # sim_score = ((similar_weight * sim) + ((1 - similar_weight) * i)) / i
        sim_score = similar_weight * sim + (1 - similar_weight) * (1 - i / len(indexes))

        # Step 4: 如果相似度更高，则更新当前桶
        if sim_score > max_similar:
            max_similar = sim_score
            best_key = hash_key

    return best_key, sim_score


def find_index_by_bucket(indexes, bucket_hash, original_lst):
    """
    查找哈希索引表中指定 hash_key 的加密向量，在原始列表中的位置
    :param indexes: LWE 哈希索引表
    :param bucket_hash: 目标 hash_key
    :param original_lst: 包含原始数据的列表
    :return: 如果找到返回匹配的索引，否则返回 -1
    """

    # 遍历索引表
    for i, (hash_key, encrypted_v) in enumerate(indexes):
        if hash_key == bucket_hash:
            # 在原始列表中寻找相应元素并返回下标
            similarity_max = 0
            j_max = -1
            for j, v in enumerate(original_lst):
                encrypted_original = lwe_encrypt_vector(v)
                similarity = np.dot(encrypted_v, encrypted_original) / (
                        np.linalg.norm(encrypted_v) * np.linalg.norm(encrypted_original))
                if similarity_max < similarity:
                    j_max = j
                    similarity_max = similarity
            return j_max


def lwe_decrypt_vector(encrypted_v, sk, pk):
    """ 对向量 encrypted_v 进行 LWE 解密 """
    A = np.array(pk[0])  # 将公钥中的矩阵 A 和 S·A + E 加载到 NumPy 数组中
    SA = np.array(pk[1])
    s = np.array(sk, dtype=np.uint16)[0:A.shape[1]]  # 裁剪sk以匹配矩阵A的列数，并将其转换为NumPy数组
    c = encrypted_v  # 将加密向量转换为 NumPy 数组
    r = (c - np.dot(A, s)) % 12289  # 计算 r = C - A·s （模 12289）
    return int_list(r)


def int_list(C):
    result = [int(c) for c in C]
    return result


# index = lwe_generate_index(data_list, num_buckets=len(data_list))

avsim = []
sumsim = 0

# for i in range(len(test_list)):
#     target_list = test_list[i]
#     bucket_hash, sim = lwe_search_index(index, target_list)
#     num = find_index_by_bucket(index, bucket_hash, data_list)
#     avsim.append(sim)
#     sumsim = sumsim + sim
#     print(sim)

datai = [0.687544935406267,
         0.6819642129541184,
         0.6880492768493294,
         0.6811395438087855,
         0.6725687137882608,
         0.6898424592318417,
         0.6796141503087227,
         0.6713569218862201,
         0.6787193489502387,
         0.6778558181877203,
         0.6781766228966197,
         0.6785725700087217,
         0.6735833675479127,
         0.6888912147971128,
         0.6882097985265475,
         0.6841474693239936,
         0.6820025288183271,
         0.6874880205713301,
         0.6822958451410971,
         0.6760127300840099,
         0.6823918281458123,
         0.6804142649096068,
         0.6706234387196741,
         0.6809297723260349,
         0.6808718887088177,
         0.6813329361385818,
         0.6774729050151524,
         0.6809449233408817,
         0.6736517926254851,
         0.6795402761903782,
         0.6874488834397003,
         0.66645902868119,
         0.6755993222691593,
         0.678954293264953,
         0.675327560593204,
         0.6909368364333056,
         0.6732181632940948,
         0.6772959004235833,
         0.6775556663714097,
         0.6821434924714367,
         0.6683429377503661,
         0.6808907833116713,
         0.6714420461266181,
         0.6727042361451743,
         0.6846164852192375,
         0.6844000709883719,
         0.6830618610968594,
         0.6872511763859214,
         0.6749657127361838,
         0.6716017848117801,
         0.6885881053285958,
         0.6771660394750851,
         0.677242643835502,
         0.6713599822169816,
         0.6741498549662199,
         0.6809434789070133,
         0.6814834042726641,
         0.6806404960636043,
         0.682730007929551,
         0.6730153116425892,
         0.6833035569012448,
         0.6820343092384525,
         0.6787174176898461]
sumdata = 0
for i in range(len(datai)):
    sumdata = sumdata + datai[i]
sumdata = sumdata / len(datai)
print(sumdata)
print(len(data_list))
print(len(test_list))

metrics_value = datai
i = 0
x = [i for i in range(len(datai))]
x = 1
plot_save_path = r'result/'
save_metrics = plot_save_path + 'fea' + '.jpg'
plt.figure()
plt.plot(datai, label=str('sim'))
plt.savefig(save_metrics)
plt.close()

sys.exit()

sumsim = sumsim / len(data_list)
avsim.sort()
print(sumsim)
print(avsim)
sys.exit()
target_list = test_list[16]
bucket_hash, sim = lwe_search_index(index, target_list)
num = find_index_by_bucket(index, bucket_hash, data_list)
print(sim)
print(num)
sys.exit()
image = Image.open("ProstateData/train/imagesTr" + "{:03d}.png".format(num))
image.show()
target_list = test_list[64]
bucket_hash, sim = lwe_search_index(index, target_list)
num = find_index_by_bucket(index, bucket_hash, data_list)
print(sim)
print(num)
image = Image.open("ProstateData/train/imagesTr" + "{:03d}.png".format(num))
image.show()
