import numpy as np
import random


## Below are some thoughts on the state, reward

# This is the current initial, which is differnt than the state that is given throughout the game
initial_state_current = {
    "player_num": 1,
    "player_count": 2,
    "position": [1,3],
    "lighthouses": [[1,1],[3,1]],
    "map": ["map for game"]
}

#This is the current state
state_current = {
    "position": [1,3],
    "energy": 66,
    "score": 36,
    "view": ["7x7 matrix with energy of 3 closest cells on all sides"],
    "lighthouses": [
        {"position": [1,1], "owner": 0, "energy": 30, "connections": [[1,3]], "have_key": False},
        {"position": [3,1], "owner": -1, "energy": 0, "connections": [], "have_key": False},
    ],
}

# I propose this state, which contains the information included in the "initial state" and removes the reward
state_proposed = {
    "player_num": 1,
    "player_count": 2,
    "position": [1,3],
    "energy": 66,
    "view": ["7x7 matrix with energy of 3 closest cells on all sides"],
    "lighthouses": [
        {"position": [1,1], "owner": 0, "energy": 30, "connections": [[1,3]], "have_key": False},
        {"position": [3,1], "owner": -1, "energy": 0, "connections": [], "have_key": False},
    ],
    "map": [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0]
            ],
}

# The reward could be separated in the reward for the current round received from taking the action and the 
# total reward for the player for the game
reward_one_round_proposed = {
    "score_current_round": 5 
}
reward_total_proposed = {
    "score": 36
}

# Not sure how the structure will be for the actions. This is my first proposal.
actions = {
    "pass": (0,0), 
    "move" : {
        "move_up": (0,1), 
        "move_down": (0,-1), 
        "move_left": (-1,0), 
        "move_right": (1,0), 
        "move_up_left": (-1,1), 
        "move_up_right": (1,1),
        "move_down_left": (-1,-1),
        "move_down_right": (-1,1),
    },
    "attack": 80, #amount = user_func(), (need to specify amount, ex. amt = max(0, 10 + energy_faro), user must define function)
    "connect": [1,1], #coords = user_func(), (location of lighthouse you want to connect with, user must define function to give this)
    }


class RandBot:
    def __init__(self, state_proposed, actions, map, env=None): # env should be specified and state_proposed removed
        # I think we should include all this information in the state that each player receives each turn
        self.env = env
        self.state = state_proposed # env.state
        self.actions = actions # env.actions
        self.lighthouses = dict((tuple(lh["position"]), lh) for lh in state_proposed["lighthouses"])
        self.position = state_proposed["position"]
        self.cx = state_proposed["position"][0]
        self.cy = state_proposed["position"][1]


    def play(self):
        """ 
        Select an action to take. If the player is on a lighthouse cell, choose if take an action.
        """
        # If on lighthouse cell, 60% probability will connect or recharge. Otherwise select a move.
        if (self.cx, self.cy) in self.lighthouses:
            if np.random.randint(0,100) < 60:
                action = self.lighthouse_action()
            else:
                action = self.choose_move() 
        else:
            action = self.choose_move()

        return action


    def choose_move(self):
        """
        Choose an action for the agent to take.
        The agent chooses an action randomly.
        
        Args:
            state: The current state of the agent.
        
        Returns:
            dict (command, x, y): The command and the move values
        """
        
        # Determinar movimientos válidos
        valid_actions = [action for action in self.actions["move"] if self.state["map"][self.cx + self.actions["move"][action][0]][self.cy + self.actions["move"][action][1]]]
        action = actions[np.random.choice(valid_actions)]
        
        return {
            "command": "move",
            "x": action[0],
            "y": action[1]
        }


    def lighthouse_action(self):
        """ 
        If player is on a lighthouse cell, choose what player should do.
        """
        # Check if there are any possible connections if the lighthouse is owned by the player
        if self.lighthouses[(self.cx, self.cy)]["owner"] == self.state.player_num:
            possible_connections = []
            for dest in self.lighthouses:
                # Do no connect with the same lighthouse
                # Do not connect if no key
                # Do not connect if the connection already exists
                # Do not connect if player does not control destination
                # Note: does not check if connection crosses
                if (dest != (self.cx, self.cy) and
                    self.lighthouses[dest]["have_key"] and
                    [self.cx, self.cy] not in self.lighthouses[dest]["connections"] and
                    self.lighthouses[dest]["owner"] == self.state["player_num"]):
                        possible_connections.append(dest)
            
            if possible_connections:
                if np.random.randint(0,100) < 60:
                    action = self.select_connect_lighthouse()
                else:
                    action = self.set_attack_recharge_energy() 
        else:
            action = self.set_attack_recharge_energy() 
        
        return action


    def set_attack_recharge_energy(self):
        """
        Select amount of energy to add to the lighthouse.

        Args:
            state: The current state of the agent

        Returns:
            dict: Command and energy to add to lighthouse
        """
        energy = np.random.randint(0, self.state["energy"] + 1)
        
        return {
            "command": "attack",
            "energy": energy
        }


    def select_connect_lighthouse(self, possible_connections):
        """
        Select the lighthouse to connect with.
        """
        destination = random.choice(possible_connections)

        return {
            "command": "connect",
            "destination": destination
        }