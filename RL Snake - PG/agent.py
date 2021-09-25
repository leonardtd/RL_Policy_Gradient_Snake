#################
#https://www.youtube.com/watch?v=PJl4iabBEz0
#################
import tensorflow as tf
import sys
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import LinearPG
from helper import plot

#Test with params
MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.01

class Agent():
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #control randomness
        self.gamma = 0.9 #discount rate: BELLMAN'S EQUATION
        self.memory = deque(maxlen=MAX_MEM) #popleft() if full

        self.optimizer = tf.keras.optimizers.Adam(1e-3)

        self.model = LinearPG(self.gamma, hidden_size=256, output_size=3)
        
    def clear_memory(self):
        self.memory = deque(maxlen=MAX_MEM)
    """
    STORE 11 VALUES: [danger straight, danger right, danger left,
                    dir left, dir right, dir up, dir down,
                    food l, food r, food u, food d]
    """
    def get_state(self, game):
        head = game.snake[0] #first item in list is head

        #Check boundaries
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x , head.y - 20)
        point_d = Point(head.x , head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        #11 values
        state = [
            #Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            #Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            #Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x < game.head.x, #Food is left of head
            game.food.x > game.head.x, #Food is right
            game.food.y < game.head.y, #Food is up
            game.food.y > game.head.y #Food is down

        ]

        return np.array(state,dtype=int)


    def get_action(self,state):
        logits = self.model.model.predict(state.reshape(1,11))
        action = tf.random.categorical(logits,num_samples=1)
        action=action.numpy()

        return action.item(0)


    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) #popleft if exceeds MAX MEM

    def train_step(self):
        states, actions, rewards, next_states, dones = zip(*self.memory)

        states = np.vstack(states)

        states = tf.convert_to_tensor(states.reshape(-1,11))

        actions = np.array(actions)      

        #sys.exit()

        self.model.train_step(self.optimizer,states,actions,rewards)
    


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        #get current state
        cur_state = agent.get_state(game)
        #get move
        action = agent.get_action(cur_state)

        #perform move and get new state
        move = [0,0,0]
        move[action] = 1
        reward, done, score = game.play_step(move)
        new_state = agent.get_state(game)

        #remember in deque
        agent.remember(cur_state,action,reward,new_state,done)

        if done:
            #train the long memory, REPLAY MEMORY: EXPERIENCED REPLAY, plot the result
            game.reset()
            
            agent.n_games += 1
            #Train agent
            agent.train_step()

            #saves new highscore
            if score > record:
                record = score
                agent.model.save()

            #Reset memory
            agent.clear_memory()

            #Print stats
            print('Game ', agent.n_games, 'Score ', score, 'Record ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score/agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()


