# import pickle as pkl
# import numpy as np
# from create_initial_states import check_game_draw, check_winner
# from tqdm import tqdm

# all_state_info = pkl.load(open("initial_states.pkl", "rb"))

# num_iterations = 5
# gamma = 0.9
# def run_value_iteration():
#     for _ in tqdm(range(num_iterations)):
#         for i, state in enumerate(tqdm(all_state_info["states"], leave=False)):
            
#             if i==29:
#                 print(state)
#             player = -1 if np.sum(state) == 0 else 1
#             #player = 1 if np.sum(state) == 0 else -1
#             player_key = "x_val" if player == -1 else "o_val"
#             opponent = (-1) * player 

#             # if state is already terminal no need to update
#             if check_game_draw(state) or check_winner(state, player) or check_winner(state, opponent):
#                 continue
            
#             allowed_moves = np.where(np.array(state) == 0)[0]
#             move_values = []
#             #iterating over all possible moves for player given state
#             for move in allowed_moves:
#                 next_state = state.copy()
#                 next_state[move] = player
#                 # after this move, other player will come into action

#                 ## Finding possible states after opponent
#                 possible_states_after_opponent = []
#                 # reversing the players because next move is for other player
#                 if check_game_draw(next_state) or check_winner(next_state, player) or check_winner(next_state, opponent):
#                     possible_states_after_opponent.append(all_state_info["states"].index(next_state)) # because if draw or winner at this state then no other player move
                
#                 else:
#                     opponent_allowed_moves = np.where(np.array(next_state) == 0)[0]
#                     for next_move in opponent_allowed_moves:
#                         next_next_state = next_state.copy()
#                         next_next_state[next_move] = opponent
#                         possible_states_after_opponent.append(all_state_info["states"].index(next_next_state))
#                         # need indices because we need each state's x and o values

#                 ## Updating the values
#                 # each possible state is equally likely to be chosen
#                 p = 1 / len(possible_states_after_opponent)
#                 move_values.append(sum([p*all_state_info["values"][player_key][s] for s in possible_states_after_opponent]))

#             all_state_info["values"][player_key][i] = gamma * max(move_values)


# run_value_iteration()
# with open("value_iteration_out.pkl", "wb") as f:
#     pkl.dump(all_state_info, f)

#######################################################################
#
# Retrieve the initialized state-value pairs from state_extractor.py
# and apply Value Iteration over all the states several times 
# until convergence (usually not more than 3 loops)
#
#######################################################################
 
import pickle

import numpy as np
import copy

def check_done(state):
    # Checks if the state is terminal (done) and 
    # returns value based on the condition of termination
    
    board_matrix = np.array(state).reshape((3, 3))
    sum_v = np.sum(board_matrix, axis=0)
    sum_h = np.sum(board_matrix, axis=1)
    checker = np.concatenate((sum_v, sum_h, 
                                [np.trace(board_matrix), 
                                np.trace(np.fliplr(board_matrix))]))

    x_wins = 3 in checker
    o_wins = -3 in checker
    draw = 0 not in state
    
    if x_wins:
        done = 1      
    elif o_wins:
        done = -1
    elif draw:
        done = 2
    else:
        done = 0
    
    return done


def possible_outcome_indices(states, state, action):
    next_state = copy.copy(state)
    player = (sum(state) == 0)
    if player:
        next_state[action] = 1
    else:
        next_state[action] = -1
    
    indices = []
    
    done = check_done(next_state)

    if done == 0:
        legal_opponent_moves = [i for i, value in enumerate(next_state) if value == 0]
        
        
        for move in legal_opponent_moves:
            possible_state = copy.copy(next_state)
            
            if player:
                possible_state[move] = -1
            else:
                possible_state[move] = 1
            indices.append(states.index(possible_state))
        
    else:
        indices.append(states.index(next_state))
    
    return indices

with open('/mnt/f/COURSE/DPRL/state_value_init.txt', 'rb') as file:
    memory = pickle.load(file)
    
states = memory[0]
x_values = memory[1]
o_values = memory[2]

delta = 5

gamma = 0.9

for i in range(5):
    delta = 0
    for state in states:
        done = check_done(state)
        index = states.index(state)
        if index == 29:
            print(state)
        player = (sum(state) == 0)  # 0 for o, 1 for x
        
        if done != 0:
            continue
        
        legal_moves = [i for i, value in enumerate(state) if value == 0]
        legal_values = []
        
        for action in legal_moves:
            outcome_indices = possible_outcome_indices(states, state, action)
            
            p = 1 / len(outcome_indices)
            action_value = 0
            
            for i in outcome_indices:
                if player:
                    value = x_values[i]
                else:
                    value = o_values[i]
                action_value += p * value          
            
            legal_values.append(round(action_value, 3))  
        
        updated_value = max(legal_values)
        
        if player:
            delta = max(delta, abs(x_values[index]-updated_value))
            x_values[index] = gamma * updated_value
        else:
            delta = max(delta, abs(o_values[index]-updated_value))
            o_values[index] = gamma * updated_value   
        
    # print(delta)

state_action = [states, x_values, o_values]

with open('state_action.txt', 'wb') as file:
    pickle.dump(state_action, file)
    
print('Successfully iterated over the states until convergence!')
print('Enjoy it!')
        
    