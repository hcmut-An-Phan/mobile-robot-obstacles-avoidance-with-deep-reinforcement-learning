import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

"""
  Plot Q-value and Reward graph
"""


def read_file(file):
    content = sio.loadmat(file)
    data = content['data']
    step = []
    Q = []

    for i in range(len(data[0])):
        if i%2 == 0:
            step.append(data[0][i])
        else:
            Q.append(data[0][i])

    return np.array(Q)


def running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-4000):(t+1)].mean()
  return running_avg

Q = read_file('step_Q.mat')
a = read_file('step_reward.mat')

Q_avg = running_avg(Q)
a_avg = running_avg(a)

plt.figure(figsize=(15, 5))  
plt.subplot(1, 2, 1)  
plt.plot(Q_avg)
plt.title("Q-value Average")
plt.xlabel("step")
plt.ylabel("Q-value")
plt.grid(True)

plt.subplot(1, 2, 2)  
plt.plot(a_avg)
plt.title("reward Average")
plt.xlabel("step")
plt.ylabel("reward")
plt.grid(True)
plt.show()



