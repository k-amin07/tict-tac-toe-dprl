# tict-tac-toe-dprl
Solve tic tac toe using reinforcement learning. 

In ttt_3d, two agents play each other as X and O, initially exploring and then using a combination of exploration and exploitation to select the actions. To improve learning, the selected action (optimal Q value from neural network during exploitation or random action during exploration) is overridden if another action can result in an immediate win or prevent an immediate loss. The agents are retrained after each game. Player O model learned to fork at some point, resulting in better game strategy.

In ttt_play, any one of the player policies can be replaced with human (manual) input to play against AI. This file was used in the final competition, where the input for human player was obtained from the AI of other participants.

The winner of the competition used a DP approach, selecting the best possible action given the current board state. ttt_train_against_dp uses the better of the two models (player O) from ttt_3d to train against the competition winner. After a 100 games, the stats are as follows


| X  | O  | Draw |
| -- | -- | ---- |
| 93 | 4  |   4  |



Credits:

[-] My group partner [Junaid Ahmed](https://github.com/Muhammad-Junaid-Ahmad).

[-] [Ahmed Muzammil](https://github.com/ahmediq-git) for providing his [competition winning code](./toUseCompetition.ipynb).