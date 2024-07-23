#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pygame

import engine, view


class CommError(Exception):
    pass


class Interface(object):
    def __init__(self, game, bots, debug=False):
        self.game = game
        self.bots = bots
        self.debug = debug

    def turn(self, player, move):
        if not isinstance(move, dict) or "command" not in move:
            raise CommError("Invalid command structure")
        try:
            if move["command"] == "pass":
                pass
            elif move["command"] == "move":
                if "x" not in move or "y" not in move:
                    raise engine.MoveError("Move command requires x, y")
                player.move((move["x"], move["y"]))
            elif move["command"] == "attack":
                if "energy" not in move or not isinstance(move["energy"], int):
                    raise engine.MoveError("Attack command requires integer energy")
                if player.pos not in self.game.lighthouses:
                    raise engine.MoveError("Player must be located at target lighthouse")
                self.game.lighthouses[player.pos].attack(player, move["energy"])
            elif move["command"] == "connect":
                if "destination" not in move:
                    raise engine.MoveError("Connect command requires destination")
                try:
                    dest = tuple(move["destination"])
                    hash(dest)
                except:
                    raise engine.MoveError("Destination must be a coordinate pair")
                self.game.connect(player, dest)
            else:
                raise engine.MoveError("Invalid command %r" % move["command"])
            return {"success": True}
        except engine.MoveError as e:
            return {"success": False, "message": str(e)}

    def get_state(self, player):
        # Lighthouses info extraction
        lighthouses = []
        for lh in self.game.lighthouses.values():
            connections = [next(l for l in c if l is not lh.pos)
                            for c in self.game.conns if lh.pos in c]
            lighthouses.append({
                "position": lh.pos,
                "owner": lh.owner,
                "energy": lh.energy,
                "connections": connections,
                "have_key": lh.pos in player.keys,
            })

        # Extract the fields for calculating the state
        player_view = self.game.island.get_view(player.pos)

        state =  {
            "position": player.pos,
            "score": player.score,
            "energy": player.energy,
            "view": player_view,
            "lighthouses": lighthouses
        }

        return state

    def estimate_reward_old(self, bot):
        """
        The logic for estimating the reward is the difference of score between
        two consecutive actions
        """
        len_scores = len(bot.scores)
        score_diff = bot.scores[-1] - bot.scores[len_scores - 2]

        return np.clip(0, 1, score_diff)
    
    def estimate_reward(self, bot, action, next_state, state):
        """
        The logic for estimating the reward is the following:
        1. 2 points for attacking a lighthouse
        2. 3 points for connecting two lighthouses
        3. 1 point for increasing the bot's energy
        4. -1 point for decreasing the bot's energy
        """
        ## TODO: Include connection between 3 lighthouses
        if action['command'] == "attack":
            return 0.75
        elif action['command'] == "connect":
            return 1
        elif next_state['energy'] > state['energy']:
            return 0.25
        else:
            return 0
    

    def run(self, max_rounds):
        game_view = view.GameView(self.game)
        round = 0
        running = True
        while round < max_rounds and running:
            # Event handler for game engine
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.game.pre_round()
            game_view.update()

            player_idx = 0
            for bot in self.bots:
                player = self.game.players[player_idx]

                ####################################################
                # If round 0, Get initial state and initialize bot
                ####################################################
                if round == 0:
                    bot.player_num = player.num
                    bot.map = self.game.island.map
                    state = self.get_state(player)
                    bot.initialize_game(state)
                else:
                    state = next_state

                ###########################################
                # Get action
                ###########################################
                action = bot.play(state)

                ###########################################
                # Execute action and get rewards and next state
                ###########################################
                status = self.turn(player, action)

                if self.debug:
                    try:
                        bot.error(status["message"], action)
                    except:
                        pass

                bot.scores.append(player.score)
                game_view.update()

                next_state = self.get_state(player)
                reward = self.estimate_reward(bot, action, next_state, state)
                transition = [state, action, reward, next_state]
                bot.transitions.append(transition)

                player_idx += 1

            self.game.post_round()

            ###########################################
            # Print the scores after each round
            ###########################################

            s = "########### ROUND %d SCORE: " % round
            for i in range(len(self.bots)):
                s += "P%d: %d " % (i, self.game.players[i].score)
            print(s)

            round += 1
                
        ###########################################
        # Optimize models
        ###########################################
        for bot in self.bots:
            bot.optimize_model(bot.transitions)
        
    