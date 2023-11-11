import pickle as pkl
import numpy as np
from create_initial_states import check_game_draw, check_winner
from tqdm import tqdm

all_state_info = pkl.load(open("initial_states.pkl", "rb"))

num_iterations = 5
gamma = 0.9
def run_value_iteration():
    for _ in tqdm(range(num_iterations)):
        for i, state in enumerate(tqdm(all_state_info["states"], leave=False)):
            
            player = -1 if np.sum(state) == 0 else 1
            #player = 1 if np.sum(state) == 0 else -1
            player_key = "x_val" if player == -1 else "o_val"
            opponent = (-1) * player 

            # if state is already terminal no need to update
            if check_game_draw(state) or check_winner(state, player) or check_winner(state, opponent):
                continue
            
            allowed_moves = np.where(np.array(state) == 0)[0]
            move_values = []
            #iterating over all possible moves for player given state
            for move in allowed_moves:
                next_state = state.copy()
                next_state[move] = player
                # after this move, other player will come into action

                ## Finding possible states after opponent
                possible_states_after_opponent = []
                # reversing the players because next move is for other player
                if check_game_draw(next_state) or check_winner(next_state, player) or check_winner(state, opponent):
                    possible_states_after_opponent.append(all_state_info["states"].index(next_state)) # because if draw or winner at this state then no other player move
                
                else:
                    opponent_allowed_moves = np.where(np.array(next_state) == 0)[0]
                    for next_move in opponent_allowed_moves:
                        next_next_state = next_state.copy()
                        next_next_state[next_move] = opponent
                        possible_states_after_opponent.append(all_state_info["states"].index(next_next_state))
                        # need indices because we need each state's x and o values

                ## Updating the values
                # each possible state is equally likely to be chosen
                p = 1 / len(possible_states_after_opponent)
                move_values.append(sum([p*all_state_info["values"][player_key][s] for s in possible_states_after_opponent]))

            all_state_info["values"][player_key][i] = gamma*max(np.round(move_values, 4))


run_value_iteration()
with open("value_iteration_out.pkl", "wb") as f:
    pkl.dump(all_state_info, f)

