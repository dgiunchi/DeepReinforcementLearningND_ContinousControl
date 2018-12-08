[![chart DDPG](https://github.com/dgiunchi/DeepReinforcementLearningND_ContinousControl/blob/master/ChartDDPG.png)](#training)

Chart of the loss function for agent using DDPG with Vector Observation space size (per agent) 33
The algorithm was DDPG, with actor and critic models. The actor networks (local and target) contains two fully connected layers with relu and final
activation function tanh that gives -1 to 1 predictions.
The critic has 3 hidden layers as fully connected.
With DDPG and fully connected model the solution came after 195 episodes. 

Following the parameters for the model:


The hyperparameters for the ddpg agent(s) are the following:

BUFFER_SIZE = int(1e6)  # replay buffer size

BATCH_SIZE = 64         # minibatch size

GAMMA = 0.99            # discount factor

TAU = 1e-3              # for soft update of target parameters

LR_ACTOR = 1e-4         # learning rate of the actor 

LR_CRITIC = 3e-4        # learning rate of the critic

WEIGHT_DECAY = 0.0001   # L2 weight decay


The DDPG algorithm belong to the category Actor-Critic algorithm, which actor that perform an action through policy evaluation while critic perform predictions through
value-based method.

The main function that runs the code is in jupyter notebooks while ddpg_agent.py contains the agent entity with step/act/learn functions.
models.py contains the different network models used in the different examples.

A replay buffer is used for store the experience as stack of states,actions, rewards, nextstates and flags if episode terminates.
This buffer is interrogated when the agent has to learn which action to take. Priority is not yet implemented and a random sample of experiences is used instead.

A noise function based on Ornstein Uhlenbeck Noise (https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) is added to action in order to increase exploration.

Models are used to predict the actions to take, both actor and critic use a local and target network as used in DQN example.
These networks are used in order to have a better estimantion and to avoid instabilities. Local and Target network are updated in different ways. Target onw a soft update parametrized 
by TAU hyperparameter, simply copying the value from the local network multiplied by that TAU and its current version of weights.
Local network instead is evaluated independently by target. 

Future work could be use random network distillation in order to explore with more efficient the environment.
