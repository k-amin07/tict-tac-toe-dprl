import numpy as np
import pickle as pkl

def check_game_draw(state):
    if np.count_nonzero(state) == 9:
        return True
    else:
        return False
def check_winner(state, player):
    win_conditions = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                    (0, 3, 6), (1, 4, 7), (2, 5, 8),
                    (0, 4, 8), (2, 4, 6)]

    for condition in win_conditions:
        if all(state[i] == player for i in condition):
            return True
    return False

curr_state = [
    0, 0, 0,
    0, 0, 0,
    0, 0, 0,
]

x_val = 0.5
o_val = 0.5
all_state_info = {
    "states": [curr_state],
    "values": {
        "x_val": [0.5],
        "o_val": [0.5],
    },
}

def create_initial_states(curr_state):

    # first move is always for X (player -> -1). 
    # So, whenver sum is zero this means one iteration of x & O
    # is complete so next is X move
    player = -1 if np.sum(curr_state) == 0 else 1
    allowed_moves = np.where(np.array(curr_state) == 0)[0]

    for move in allowed_moves:
        next_state = curr_state.copy()
        next_state[move] = player
        
        if next_state in all_state_info["states"]:
            continue
        
        
        if check_winner(next_state, player): 
            all_state_info["values"]["x_val"].append(player*(-1))
            all_state_info["values"]["o_val"].append(player)
        elif check_game_draw(next_state):
            all_state_info["values"]["x_val"].append(0)
            all_state_info["values"]["o_val"].append(0)
        else:
            create_initial_states(next_state)
            all_state_info["values"]["x_val"].append(0.5)
            all_state_info["values"]["o_val"].append(0.5)
            
        all_state_info["states"].append(next_state) 


if __name__ == "__main__":
    create_initial_states(curr_state)
    print(len(all_state_info["states"]))

    with open("initial_states.pkl", "wb") as f:
        pkl.dump(all_state_info, f)