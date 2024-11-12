#!/usr/bin/python
# -*- coding: utf-8 -*-

import engine, train as train
import pandas as pd
import os

from bots.ppo import PPO
from bots.randbot import RandBot


# ==============================================================================
# MAIN
# Main process for simulating matches of different types of bots
# ==============================================================================

if __name__ == "__main__":
    # Set the map
    cfg_files_train = ["./maps/game_map.txt"] #, "maps/island.txt", "maps/random.txt"] #, "maps/square_xl.txt"]
    cfg_file_eval = "./maps/game_map.txt"
    # Set the bots to play the game

    # REINFORCE(state_maps=False, trained_model_filename='reinforce_mlp.pth', save_model_filename = "reinforce_mlp.pth", use_saved_model=True),
    #         REINFORCE(state_maps=False, trained_model_filename=None, save_model_filename = None, use_saved_model=False),
    #         REINFORCE(state_maps=False, trained_model_filename=None, save_model_filename = None, use_saved_model=False),

    NUM_EPISODES = 10
    MAX_AGENT_UPDATES = 90 # Number of times to update the agent within an episode
    NUM_STEPS_POLICY_UPDATE = 128 # Number of experiences to collect for each update to the agent
    MAX_EVALUATION_ROUNDS = 1000
    TRAIN = True # Whether to run training or evaluation
    NUM_ENVS = 1
    STATE_MAPS = True
    MODEL_FILENAME = "ppo_transfer_cnn_1envs_new_rew.pth"
    USE_SAVED_MODEL = False
    MAX_TOTAL_UPDATES = NUM_EPISODES * MAX_AGENT_UPDATES

    #######################################################################
    # Total number of rounds = MAX_AGENT_UPATES * NUM_STEPS_POLICY_UPDATE #
    #######################################################################

    #bots = [REINFORCE(state_maps=False, model_filename = "reinforce.pth", use_saved_model=True),]
    bots = [PPO(state_maps=STATE_MAPS, num_envs=NUM_ENVS, num_steps=NUM_STEPS_POLICY_UPDATE, num_updates=MAX_TOTAL_UPDATES, train=TRAIN, model_filename = MODEL_FILENAME, use_saved_model=USE_SAVED_MODEL),]

    if TRAIN:
        for i in range(1, NUM_EPISODES+1):
            for i in range(len(cfg_files_train)):
                config = engine.GameConfig(cfg_files_train[i])
                game = [engine.Game(config, len(bots)) for i in range(NUM_ENVS)]
            # config = engine.GameConfig(cfg_file)
            # game = [engine.Game(config, len(bots)) for i in range(NUM_ENVS)]

                iface = train.Interface(game, bots, debug=False)
                iface.train(max_updates=MAX_AGENT_UPDATES, num_steps_update=NUM_STEPS_POLICY_UPDATE)
            # iface = train_ppo.Interface(game, bots, debug=False)
            # iface.train(max_updates=MAX_AGENT_UPDATES, num_steps_update=NUM_STEPS_POLICY_UPDATE)
    
    if not TRAIN:
        for i in range(1, NUM_EPISODES+1):
            config = engine.GameConfig(cfg_file_eval)
            game = [engine.Game(config, len(bots))]

            iface = train.Interface(game, bots, debug=False)
            iface.run(max_rounds=MAX_EVALUATION_ROUNDS)
            final_scores_list = []
            for bot in bots:
                bot.final_scores_list.append(bot.scores[-1][0])
        
        final_scores = pd.DataFrame()
        for bot in bots:
                final_scores["bot_"+str(bot.player_num)] = bot.final_scores_list
        os.makedirs('./final_scores', exist_ok=True)
        final_scores.to_csv('./final_scores/final_scores_cnn_1env_extra_connect3_update.csv', index_label='episode')



    




       