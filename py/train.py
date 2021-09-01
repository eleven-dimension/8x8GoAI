import os
import time
import numpy as np

from net import *

def get_next_code(s):
    return "{0:0>6}".format(int(s) + 1)
        
def get_newest_code(path):
    with open(path, "r") as f:
        text = f.read()
    return text

ROTATE_NUM = 1

WIDTH = 8

BLACK = 1
WHITE = -1

BATCH_SIZE = 512
STEPS = 100

t = time.process_time()

game_path = "../games"
g = os.walk(game_path)

tot_line_cnt = 0
game_line_map = {}
winner_map = {}

for path, dir_list, file_list in g:
    for dir_name in dir_list:
        with open(path + '\\' + dir_name + '\length.txt') as f:
            game_line_cnt = int(f.read())
        with open(path + '\\' + dir_name + '\winner.txt') as f:
            winner = int(f.read())
        tot_line_cnt += game_line_cnt * ROTATE_NUM
        game_line_map[dir_name] = game_line_cnt
        winner_map[dir_name] = winner

print(tot_line_cnt)

x_train = np.zeros((tot_line_cnt, 3, WIDTH, WIDTH))
y_prob = np.zeros((tot_line_cnt, WIDTH * WIDTH + 1))
y_win = np.zeros((tot_line_cnt, 1))

line_index = 0
g = os.walk(game_path)

for path, dir_list, file_list in g:
    for dir_name in dir_list:
        lines = game_line_map[dir_name]
        if lines == 0:
            print(dir_name)
            continue
        winner = winner_map[dir_name]
        
        x = np.loadtxt(path + '\\' + dir_name + '\\in.txt', dtype=np.float32)
        x = x.reshape((lines, 3, WIDTH, WIDTH))
        prob = np.loadtxt(path + '\\' + dir_name + '\\prob.txt', dtype=np.float32)
        prob = prob.reshape((lines, WIDTH * WIDTH + 1))
        
        # print(path + '\\' + dir_name + '\\value.txt')
        search_value = np.loadtxt(path + '\\' + dir_name + '\\value.txt', dtype=np.float32)
        search_value = search_value.reshape((lines, 1))
        
        for l in range(lines):
            # in
            x_train[line_index] = x[l]
            # prob
            y_prob[line_index] = prob[l]
            # win
            if winner == BLACK:
                if l % (2*WIDTH) <= (WIDTH - 1):
                    y_win[line_index][0] = (1 + search_value[l][0]) / 2.0
                else:
                    y_win[line_index][0] = (-1 + search_value[l][0]) / 2.0
            elif winner == WHITE:
                if l % (2*WIDTH) <= (WIDTH - 1):
                    y_win[line_index][0] = (-1 + search_value[l][0]) / 2.0
                else:
                    y_win[line_index][0] = (1 + search_value[l][0]) / 2.0   

            line_index += 1

state_batch = x_train.tolist()
p_batch = y_prob.tolist()
v_batch = y_win.tolist()
example_buffer = list(zip(state_batch, p_batch, v_batch))

elapsed_time = time.process_time() - t

print(elapsed_time)

# build net
net = NeuralNetWorkWrapper(lr=0.001, l2=0.0001, num_layers=8, num_channels=64, n=WIDTH, action_size=WIDTH*WIDTH + 1)

last_model_name = get_newest_code('../index/newest.txt')
net.load_model('../models', last_model_name)
net.train(example_buffer, BATCH_SIZE, STEPS)

new_model_name = get_next_code(last_model_name)
net.save_model('../models', new_model_name)

with open('../index/newest.txt','w') as f:
    f.write(new_model_name)

print("saved")                           