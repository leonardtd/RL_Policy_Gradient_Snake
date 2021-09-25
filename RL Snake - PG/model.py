#LEO
import tensorflow as tf
import os
import numpy as np
import sys
class LinearPG():
    def __init__(self, gamma, hidden_size, output_size):

        self.gamma = gamma
        self.model = tf.keras.models.Sequential([
            # First Dense layer
            tf.keras.layers.InputLayer(input_shape=(11,)),
            tf.keras.layers.Dense(units=hidden_size,activation='relu'),
            tf.keras.layers.Dense(units=output_size, activation=None) 
        ])


    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path,file_name)

        #Save model
        self.model.save(file_name)

    def train_step(self, optimizer,states, actions, rewards):
        #Discount rewards
        discounted_rewards = self.calc_discounted_rewards(rewards)
        #Calculate logits

        #EJECUTAR FORWARD EN GRADIENT TAPE
        with tf.GradientTape() as tape:
            logits = self.model(states)
            #Calculate loss
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
            loss = tf.reduce_mean(xentropy*discounted_rewards)
            #Calculate grads
        
            grads = tape.gradient(loss,self.model.trainable_variables)

            #Backprop
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def calc_discounted_rewards(self,rewards):
        R = 0
        discounted_rewards = np.zeros_like(rewards,dtype=np.float32)
        for t in reversed(range(0,len(rewards))):
            R = R*self.gamma + rewards[t]
            discounted_rewards[t] = R
        return self.normalize(discounted_rewards)

    def normalize(self,x):
        x -= np.mean(x)
        x /= np.std(x)
        return x.astype(np.float32)

        

    




        




        

    