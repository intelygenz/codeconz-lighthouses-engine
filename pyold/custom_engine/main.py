#!/usr/bin/python
# -*- coding: utf-8 -*-

import engine, train
import pandas as pd
import os

from bots.randbot import RandBot
from bots.reinforce import REINFORCE


# ==============================================================================
# MAIN
# Main process for simulating matches of different types of bots
# ==============================================================================

if __name__ == "__main__":
    # Set the map
    cfg_file = "maps/grid.txt"
    
    # Set the bots to play the game
    # bots = [REINFORCE(state_maps=False, trained_model_filename='reinforce_mlp.pth', use_saved_model=True), 
    #         REINFORCE(state_maps=True, trained_model_filename='reinforce_cnn.pth', use_saved_model=True), 
    #         RandBot()]

    bots = [REINFORCE(state_maps=True, trained_model_filename='best_reinforce_cnn.pth', save_model_filename = "reinforce_cnn.pth", use_saved_model=True)]

    NUM_TRAINING_EPISODES = 25
    MAX_AGENT_UPDATES = 5000000 # Number of times to update the agent 
    NUM_STEPS_POLICY_UPDATE = 256 # Number of experiences to collect for each update to the agent

    #######################################################################
    # Total number of rounds = MAX_AGENT_UPATES * NUM_STEPS_POLICY_UPDATE #
    #######################################################################

    for i in range(1, NUM_TRAINING_EPISODES+1):
        config = engine.GameConfig(cfg_file)
        game = engine.Game(config, len(bots))

        iface = train.Interface(game, bots, debug=False)
        iface.run(max_updates=MAX_AGENT_UPDATES, num_steps_update=NUM_STEPS_POLICY_UPDATE)


       