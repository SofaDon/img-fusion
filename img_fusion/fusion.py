from PIL import Image
from scipy.sparse import linalg as linalg
from scipy.sparse import lil_matrix as lil_matrix
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

'''
src_img = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image fusion/test1_src.jpg'
mask_img = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image fusion/test1_mask.jpg'
target_img = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image fusion/test1_target.jpg'
DEL_X = -100
DEL_Y = -40
'''
src_img = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image fusion/test2_src.png'
mask_img = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image fusion/test2_mask.png'
target_img = 'D:/大三下/数字图像处理/大作业1/第一次大作业/image fusion/test2_target.png'
DEL_X = -180
DEL_Y = 140


src = np.array(Image.open(src_img))
mask = np.array(Image.open(mask_img))
target = np.array(Image.open(target_img))

#定义的一些常数
IN_OMEGA = 0
DELTA_OMEGA = 1
OUT_OMEGA = 2

#判断某个点的位置
def judge_location(index, mask):
	if (in_omega(index, mask) == False):
		return OUT_OMEGA
	if (at_edge(index, mask) == True):
		return DELTA_OMEGA
	return IN_OMEGA

#判断某个点是否在Omega里面
def in_omega(index, mask):
	if (mask[index] == 1):
		return True
	return False

#判断某个点是否在DELTA_OMEGA上
def at_edge(index, mask):
	if (in_omega(index, mask) == False) :
		return False
	for i in get_around(index):
		if (in_omega(i, mask) == False):
			return True
	return False

#获取某个点上下左右的点
def get_around(index):
	i, j = index
	points = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
	return points

#计算散度
def caculate_div(source, index):
	i, j = index
	div = (4 * source[i, j]) - (1 * source[i-1, j]) - (1 * source[i+1, j]) - (1 * source[i, j-1]) - (1 * source[i, j+1])
	return div

#将mask中的非零元的坐标利用zip存储
def mask_positon(mask):
	nonzero = np.nonzero(mask)
	return zip(nonzero[0], nonzero[1])

#获取稀疏矩阵 即Ax=b中的系数矩阵A
def get_sparse_matrix(points):
	pn = len(points)
	#利用lil_matrix高效构建矩阵
	A = lil_matrix((pn, pn))
	#利用enumerate能够同时获取下标和坐标
	for i, index in enumerate(points):
		A[i, i] = 4
		for ii in get_around(index):
			if ii not in points :
				continue
			#获取ii这个点在points里的下标
			j = points.index(ii)
			A[i, j] = -1
	print("finish")
	return A

#Main函数
def main(source, target, mask):
	indices = mask_positon(mask)
	indices = list(indices)
	pn = len(indices)
	A = get_sparse_matrix(indices)
	b = np.zeros(pn)

	for i, index in enumerate(indices):
		b[i] = caculate_div(source, index)
		#DELTA_OMEGA上的点需要特殊处理
		if (judge_location(index, mask) == DELTA_OMEGA):
			for ii in get_around(index):
				if (in_omega(ii, mask) == False):
					iii = (ii[0]+DEL_X, ii[1]+DEL_Y)
					b[i] += target[iii]

	#print("开始计算x")
	#调函数解稀疏矩阵
	x = linalg.cg(A, b)
	#print(x)
	#复制target
	final_target = np.copy(target).astype(int)
	#根据x修改相应的像素点
	for i, index in enumerate(indices):
		indexx = (index[0]+DEL_X, index[1]+DEL_Y)
		final_target[indexx] = x[0][i]
	return final_target

if __name__=="__main__":
	mask = np.atleast_3d(mask).astype(np.float) / 255.
	mask[mask != 1] = 0
	mask = mask[:,:,0]
	channels = 3
	result_stack = [main(src[:,:,i], target[:,:,i], mask) for i in range(channels)]
	result = cv2.merge(result_stack)
	target = result


plt.figure("test")
plt.imshow(target)
plt.axis('off')
plt.show()
