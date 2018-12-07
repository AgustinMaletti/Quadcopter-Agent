import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, reward_points=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        # action repeat choice
        self.action_repeat = 3
        
        # Four  bushless motor :)
        self.action_size = 4
        # 6 states, x,y,z, and angles phi, theta, psi, 18 states becouse of the action repeat choice
        self.state_size = self.action_repeat * 6
        
        # Range of motor power
        self.action_low = 0
        self.action_high = 900
        self.motors_max = np.array([900.,900.,900.,900.])
        self.motors_min = np.array([0.,0.,0.,0.])
        
        self.init_pose = np.array(init_pose)
        #self.target_pos=np.array(target_pos)
        # Goal
        self.reward_points = np.array(reward_points)
        # Set to zero the reference points
        self.reference_of_touch_points = np.zeros_like(self.reward_points)
        
        self.target_pos = target_pos if target_pos is not None else np.array([20., 20., 40.]) 
        
    def get_reward(self, rotor_speeds):
        """Uses current pose  and the force in the motors to return reward."""
        # Normalizing distance
        norm_factor_distance = np.sqrt((self.init_pose[:3]- self.target_pos)**2).sum()
        
        distance = 1. -  abs(np.array(self.sim.pose[:3]) - self.target_pos).sum()/norm_factor_distance 
        # Normalizing motors actions
        #motor_force = (np.array(rotor_speeds) - np.array([0.,0.,0.,0.])).sum()/ (self.motors_max-self.motors_min).sum()
        
        # Down is simple reward function normalized, I' m giving a nose to the dog. The ideia is that the drone feel the distance and save motor power.
        # If the distance from target point is minor then  gain more reward, if the motor force used is minor  then gain more reward 
        # With the second one I traing to prevent sudden moviment from the drone.
        reward = (((1.- distance)*25) 
        #+ (1.- motor_force)*0.5) /2.
        
        # Checking reward points: task 1: is with motor force and reward debuff included. task 2: is without.
        for  i in range(len(self.reward_points)):
        # Check if all coordenates in the position are equal to any of the rewards points
             if np.alltrue(self.sim.pose == self.reward_points[i]):
        # Check if the drone already touch the point, only reward first time touch        
                 if  self.reference_of_touch_points[i].all() == 0:  
        # Give reward  to the drone
                     reward += 100
        #  Mark the point as touch
                     self.reference_of_touch_points[i] = 1
        # Others points punish him
        #         else:
        # Debuff    
        #          reward -= 0.025
             
        return reward
    
    
    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            # Pass the rotor speed information to the reward function
            reward += self.get_reward(rotor_speeds) 
            # Update the position
            pose_all.append(self.sim.pose)
            
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    
    def reset(self):
        """Reset the sim to start a new episode."""
        # Set all reference  of  touch points reward to zero
        self.reference_of_touch_points = np.zeros_like(self.reward_points)
        self.sim.reset()
        # Podria poner mas informaciones del sim aqui
        state = np.concatenate( [self.sim.pose] * self.action_repeat) 
        return state
    

