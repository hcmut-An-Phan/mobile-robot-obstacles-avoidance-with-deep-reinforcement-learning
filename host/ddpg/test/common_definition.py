NUM_MISSION = 40

NUM_MAX_STEP = 1500
SAVE_REWARD = False

TEST_MAZE = True
USER_INPUT = False

REAL_TEST = False
TARGET_X_CORR = 9.0
TARGET_Y_CORR = 2.0

VEL_THRESH_HOLD = False
LINEAR_THRESH_HOLD = 0.15
ANGUALR_THRESH_HOLD = 0.7

# MODEL_PATH = 'maze_per1_30k'
# MODEL_PATH = 'maze_per1_108k'
MODEL_PATH = 'actor_model-1000-108506'

# MODEL_PATH = 'maze_per2_90k'

# MODEL_PATH = 'maze_human_190ep'

# MODEL_PATH = 'maze_org_120ep'
# MODEL_PATH = 'maze_org_200ep'

# MODEL_PATH = 'big_corridor_89k'
# MODEL_PATH = 'big_corridor_100k'

# MODEL_PATH = 'small_corridor_80k'
# MODEL_PATH = 'small_corridor_86k'
# MODEL_PATH = 'small_corridor_93k'

"""

=========================================================================================================================================
=                                                       20 random test at maze                                                          =
=========================================================================================================================================


PER2: 
    ep 90k:   Finish 20 missions with 0 collision in 9345 steps, time: 0:16:55

PER1:

ORG_human:

ORG:

=========================================================================================================================================
=                                                       40 fix test at test_maze                                                        =
=========================================================================================================================================


PER2: 
    ep 200ep-90k: Finish 40 missions with 0 collision in 19834, steps, time: 0:36:02


PER1:
    ep 240ep-108k: Finish 40 missions with 0 collision in 17754 steps, time: 0:32:21


ORG_human:
    ep 190: Finish 40 missions with 2 collision in 18568 steps, time: 0:33:42


ORG:
    ep 290ep-100k: Finish 40 missions with 28 collision in 12372 steps, time: 0:22:38
    ep 150: Finish 40 missions with 17 collision in 18340 steps, time: 0:34:45
    ep 120: Finish 40 missions with 12 collision, time out 1, in 14614 steps, time: 26:32
    ep 110: Finish 40 missions with 8  collision, time out 9  in 19449 steps, time:0:35:28
    
    
=========================================================================================================================================
=                                                       20 random test at test_big_corridor                                             =
=========================================================================================================================================


PER1: 
	ep 300ep-89k: Finish 20 missions with 0 collision in 9034 steps, time 0:16:17
	human: 


=========================================================================================================================================
=                                                       20 random test at real_corridor                                             =
=========================================================================================================================================


PER1:
    300ep-89k: Finish 20 missions with 0 collision with 9136 steps in 0:17:18


"""