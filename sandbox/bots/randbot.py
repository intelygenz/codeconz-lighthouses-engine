#!/usr/bin/python
# -*- coding: utf-8 -*-
# Adapted from https://github.com/marcan/lighthouses_aicontest

import random

from bots import bot


class RandBot(bot.Bot):
    """Bot that executes random actions"""
    NAME = "RandBot"

    def play(self, state, step=None):
        cx, cy = state["position"]
        lighthouses = dict((tuple(lh["position"]), lh)
                            for lh in state["lighthouses"])
        # If there is a lighthouse in the current position
        if (cx, cy) in lighthouses.keys():
            # Probability 60%: connect to valid remote lighthouse
            if lighthouses[(cx, cy)]["owner"] == self.player_num:
                if random.randrange(100) < 60:
                    possible_connections = []
                    for dest in lighthouses.keys():
                        # Do not connect to itself
                        # Do not connect if there is no key available
                        # Do not connect if there is an existing connection
                        # Do not connect if the destination is not controlled
                        if (dest != (cx, cy) and
                            lighthouses[dest]["have_key"] and
                            [cx, cy] not in lighthouses[dest]["connections"] and
                            lighthouses[dest]["owner"] == self.player_num):
                            possible_connections.append(dest)

                    if possible_connections:
                        return self.connect(random.choice(possible_connections))

            # Probability 60%: recharge lighthouse
            if random.randrange(100) < 60:
                energy = random.randrange(state["energy"] + 1)
                return self.attack(energy)

        # Random move
        moves = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1))
        # Check if the move is valid
        moves = [(x,y) for x,y in moves if self.map[cy+y][cx+x]]
        move = random.choice(moves)

        return self.move(*move)
