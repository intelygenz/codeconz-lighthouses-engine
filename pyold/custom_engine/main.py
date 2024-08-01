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

    # REINFORCE(state_maps=False, trained_model_filename='reinforce_mlp.pth', save_model_filename = "reinforce_mlp.pth", use_saved_model=True),
    #         REINFORCE(state_maps=False, trained_model_filename=None, save_model_filename = None, use_saved_model=False),
    #         REINFORCE(state_maps=False, trained_model_filename=None, save_model_filename = None, use_saved_model=False),

    bots = [REINFORCE(state_maps=True, trained_model_filename=None, save_model_filename = None, use_saved_model=False)]

    NUM_EPISODES = 25
    MAX_AGENT_UPDATES = 5000000 # Number of times to update the agent 
    NUM_STEPS_POLICY_UPDATE = 256 # Number of experiences to collect for each update to the agent
    MAX_EVALUATION_ROUNDS = 10000
    TRAIN = False # Whether to run training or evaluation

    #######################################################################
    # Total number of rounds = MAX_AGENT_UPATES * NUM_STEPS_POLICY_UPDATE #
    #######################################################################

    for i in range(1, NUM_EPISODES+1):
        config = engine.GameConfig(cfg_file)
        game = engine.Game(config, len(bots))

        iface = train.Interface(game, bots, debug=False)
        if TRAIN:
            iface.train(max_updates=MAX_AGENT_UPDATES, num_steps_update=NUM_STEPS_POLICY_UPDATE)
        else:
            iface.run(max_rounds=MAX_EVALUATION_ROUNDS)
            final_scores_list = []
            for bot in bots:
                bot.final_scores_list.append(bot.scores[-1])
    
    if not TRAIN:
        final_scores = pd.DataFrame()
        for bot in bots:
                final_scores["bot_"+str(bot.player_num)] = bot.final_scores_list
        os.makedirs('./final_scores', exist_ok=True)
        final_scores.to_csv('./final_scores/final_scores_cnn_no_trained.csv', index_label='episode')



    




       