from keras.initializers import glorot_normal

# brain parameters
GAMMA = 0.99  # for the temporal difference
RHO = 0.001  # to update the target networks
KERNEL_INITIALIZER = glorot_normal()

# buffer params
UNBALANCE_P = 0.8  # newer entries are prioritized
BUFFER_UNBALANCE_GAP = 0.5

# training parameters
STD_DEV = 0.2
BATCH_SIZE = 512
BUFFER_SIZE = 100000
PARTITION_NUM = 100
TOTAL_EPISODES = 10000
TOTAL_STEP = 100000
MAX_STEP = 750
CRITIC_LR = 1e-4
ACTOR_LR = 1e-4
WARM_UP = 3  # num of warm up epochs
EPS = 0.9