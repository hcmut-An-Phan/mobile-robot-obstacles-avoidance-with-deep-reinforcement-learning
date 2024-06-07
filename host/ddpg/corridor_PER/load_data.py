import numpy as np
import scipy.io as sio

"""
	Test load human data
"""

mat_contents = sio.loadmat('human_data_1000.mat')
data = mat_contents['data']

print("data length is %s", len(data))


for i in range(1000):
	# print(f"data {i} len(data): {len(data[i])}")
	# print(f"data {i} current state is {data[i][0:28]}")
	print(f"data {i} action is {data[i][28]:.2f}, {data[i][29]:.2f}")
	# print(f"data {i} reward is {data[i][30]}:.2f")
	# print(f"data {i} new state is  {data[i][31:59]}")
	# print(f"data {i} done is  {data[i][59]}")
	# print(f"data {i} total is  {data[i]:.2f}")
