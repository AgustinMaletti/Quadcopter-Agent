import random
from collections import namedtuple, deque
from keras import layers, models, optimizers
from keras import backend as K
import numpy as np
import copy

# Agent
class DDPG():
    ''' Reinforcement Learning agent that learns using DDPG'''
    def __init__(self, task,
                         layer_actor, layer_critic,
                         agent_noise_mu_theta_sigma, 
                         agent_gamma_tau ):
    
        self.task = task
        self.state_size  = task.state_size
        self.action_size = task.action_size
        self.action_low  = task.action_low
        self.action_high = task.action_high
                                        
                                        
                                        
        # Actor (policy) Model
        self.actor_local  = Actor(self.state_size, self.action_size, self.action_low, self.action_high, layer_actor )
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, layer_actor)
        # Critic (Value) model
        self.critic_local  = Critic(self.state_size, self.action_size, layer_critic)
        self.critic_target = Critic(self.state_size, self.action_size, layer_critic)
        # Initialize target model parameters  with local parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())
        
        # Noise process
        self.exploration_mu = agent_noise_mu_theta_sigma[0]
        self.exploration_theta = agent_noise_mu_theta_sigma[1]
        self.exploration_sigma = agent_noise_mu_theta_sigma[2]
        
        self.noise = OUNoise(self.action_size, self.exploration_mu, 
                             self.exploration_theta, self.exploration_sigma)
        self.buffer_size = 10000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Algorith parameters
        self.gamma = agent_gamma_tau[0]
        self.tau = agent_gamma_tau[1]
    
  
    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state
    
    def step(self, action, reward, next_state, done):   
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)
        # Learn if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        # Roll over last state action
        self.last_state = next_state
    
    def act(self, state):
        '''Return actions for given  state as per current policy '''
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())
        
    def learn(self, experiences):
        ''' Update policy and value parameters usign batch of experience tuples'''
        # Convert experience tuples to separate arrays for each element( state, action, reward, etc)
        states      = np.vstack([e.state for e in experiences if e is not None])
        actions     = np.array([e.action for e in experiences if e is not None
                               ]).astype(np.float32).reshape(-1,self.action_size)
        rewards     = np.array([e.reward for e in experiences if e is not None
                               ]).astype(np.float32).reshape(-1,1)
        dones       =  np.array([e.done for e in experiences if e is not None
                                ]).astype(np.uint8).reshape(-1,1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        
        # Get predicted next_state action and Q values from target models
        action_next    = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, action_next])
        
        # Compute Q targets for current states and train critic model(local)
        Q_target =  rewards + self.gamma* Q_targets_next *(1-dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets_next)    
        
        # Train the model(local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions,0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1]) # Custom training function
        
        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)
        
    def soft_update(self, local_model, target_model):
        ''' Soft update model parameters'''
        local_weights  = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights()) 
        assert len(local_weights) == len(target_weights),'Local and target models paramenters must have the same size'
        
        new_weights = self.tau*local_weights + (1-self.tau)*target_weights
        
        target_model.set_weights(new_weights)



class Actor:
    ''' Actor(Policy) Model'''
    
    def __init__(self, state_size, action_size, action_low, action_high, layer_actor):
        ''' Initialize parameters and build model.  
        Params:
        state_size(int): Dimension of each state
        action_size(int): Dimension of each action
        action_low(array): Min value of each action dimension
        action_high(array): Max value of each action dimension'''
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high- self.action_low
        self.layer_actor = layer_actor
        self.build_model()
    def build_model(self):
        ''' Build an actor(policy) network that maps states -> actions'''
            # Define input layer
        states =  layers.Input(shape=(self.state_size,), name='states')
        # Add hidden layers
        net = layers.Dense(units=self.layer_actor[0],  activation='tanh')(states)
        net = layers.Dense(units=self.layer_actor[1],  activation='tanh')(net)
        net = layers.Dense(units=self.layer_actor[2],  activation ='tanh')(net)
        
        #Output layer
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
                                                    name='raw_actions')(net)
        actions = layers.Lambda(lambda x:(x* self.action_range) + self.action_low, name='actions')(raw_actions)
            
        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)
        # Define loss function  using action value( Q value)  gradient
        action_gradients = layers.Input(shape=(self.action_size,))
        
        loss = K.mean(-action_gradients*actions)
        # Incorporate any additional losses here
        # Define optimizer and training function
        optimizer     = optimizers.Adam()
        # optimizer in action
        updates_op  = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        
        self.train_fn = K.function(inputs=[self.model.input, action_gradients, K.learning_phase()],
                                                    outputs=[], updates = updates_op)
    
#import sys
#sys.path.append('/home/agustinmaletti/anaconda3/lib/python3.6/site-packages')
class Critic:
    ''' Critic (value) Model'''
    def __init__(self, state_size, action_size, layer_critic):
        ''' Initialize parameters and build the model
        Params:
        state_size(int): Dimension of each state
        action_size(int): Dimension of each action'''
        self.state_size = state_size
        self.action_size = action_size
        self.layer_critic = layer_critic
        # Initialize any other variable here
        self.build_model()
    
    def build_model(self):
        ''' Build a critic(value) network that maps(state,action) pairs -> Q-values'''
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions =  layers.Input(shape=(self.action_size,), name='actions')
        # add hidden layers for states pathwway
        net_states = layers.Dense(units=self.layer_critic[0], activation='tanh')(states)
        net_states = layers.Dense(units=self.layer_critic[1], activation='tanh')(net_states)
        # add hidden layers for action pathwway
        net_actions = layers.Dense(units=self.layer_critic[2], activation='tanh')(actions)
        net_actions = layers.Dense(units=self.layer_critic[3], activation='tanh')(net_actions)
        
        # conbine action and states pathway
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('sigmoid')(net)
        # add a final output  layer  to produce  action values ( Q_values)
        Q_values =  layers.Dense(units=1, name='q_values')(net)
        # Condense de model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        # choose the optimizer
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        
        # Compute action gradient(derivative  of q_values w.r.t.  actions)
        action_gradients = K.gradients(Q_values, actions)
        
        # Define an additional function to fetch action gradients( to be used by  actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], 
                                                                        outputs=action_gradients)
    
class ReplayBuffer:
    ''' Fixed-size buffer to store experience tuples'''
    
    def __init__(self, buffer_size, batch_size):
        self. memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', 
                                      field_names=['state', 'action', 
                                      'reward', 'next_state', 'done'])
    
    def add(self, state, action, reward, next_state, done):
        ''' Add a new experience to memory'''
        e =  self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size=64):
        '''Randomly sample a batch of experieces from memory'''
        return random.sample(self.memory, k=self.batch_size)
    
    def __len__(self):
        ''' Return the current size of internal memory'''
        return len(self.memory)
        
# Ornstein- Uhlenbeck process
class OUNoise:
    ''' Ornstein-Uhlenbeck process'''
    def __init__(self, size, mu, theta, sigma):
        ''' Inititalize  parameters and noise process'''
        self.mu = mu*np.ones(size) 
        self.theta = theta
        self.sigma =  sigma
        self.reset()
        
    def reset(self):
        ''' Reset the internal state (=noise) to mean(mu')'''
        self.state = copy.copy(self.mu)
    
    def sample(self):
        ''' Update internal state and return it as a noise sample'''
        x   = self.state
        dx = self.theta *(self.mu -x) + self.sigma * np.random.randn(len(x))
        self.state = x +dx
        
        return self.state



