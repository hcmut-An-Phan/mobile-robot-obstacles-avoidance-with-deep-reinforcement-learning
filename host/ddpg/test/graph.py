import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

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
    running_avg[t] = totalrewards[max(0, t-1000):(t+1)].mean()
  return running_avg

a = read_file('per1_1000step_reward.mat')

a_avg = running_avg(a)

plt.plot(a_avg)
plt.title("reward Average")
plt.xlabel("step")
plt.ylabel("reward")
plt.grid(True)
plt.show()



