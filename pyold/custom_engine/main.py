#!/usr/bin/python
# -*- coding: utf-8 -*-

import engine, interface

from randbot import RandBot
from reinforce import REINFORCE

import ipdb

# ==============================================================================
# MAIN
# Main process for simulating matches of different types of bots
# ==============================================================================

if __name__ == "__main__":
    # Set the map
    cfg_file = "maps/grid.txt"
    
    # Set the bots to play the game
    #bots = [RandBot(), RandBot(), RandBot()]
    bots = [REINFORCE(), RandBot()]

    config = engine.GameConfig(cfg_file)
    game = engine.Game(config, len(bots))

    iface = interface.Interface(game, bots)
    iface.train_reinforce(n_training_episodes=10, max_t=100,)
    #iface.run()
