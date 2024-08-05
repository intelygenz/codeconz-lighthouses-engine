#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pygame
import pandas as pd
import os

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
                if player.pos not in self.game[0].lighthouses:
                    raise engine.MoveError("Player must be located at target lighthouse")
                self.game[0].lighthouses[player.pos].attack(player, move["energy"])
            elif move["command"] == "connect":
                if "destination" not in move:
                    raise engine.MoveError("Connect command requires destination")
                try:
                    dest = tuple(move["destination"])
                    hash(dest)
                except:
                    raise engine.MoveError("Destination must be a coordinate pair")
                self.game[0].connect(player, dest)
            else:
                raise engine.MoveError("Invalid command %r" % move["command"])
            return {"success": True}
        except engine.MoveError as e:
            return {"success": False, "message": str(e)}

    def get_state(self, player, i):
        # Lighthouses info extraction
        lighthouses = []
        for lh in self.game[i].lighthouses.values():
            connections = [next(l for l in c if l is not lh.pos)
                            for c in self.game[i].conns if lh.pos in c]
            lighthouses.append({
                "position": lh.pos,
                "owner": lh.owner,
                "energy": lh.energy,
                "connections": connections,
                "have_key": lh.pos in player.keys,
            })

        # Extract the fields for calculating the state
        player_view = self.game[i].island.get_view(player.pos)

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
    
    def estimate_reward(self, action, state, next_state, player, status):
        """
        The logic for estimating the reward is the following:
        1. if "status" is False: -1
        2. if "move" and the move is invalid: -1
        3. if "move" and land on lighthouse that do not own: 0.5
        4. if "move" and increase bot's energy: 0.3
        5. if "attack" and not on a lighthouse: -1
        6. if "attack" and gain control of lighthouse: 0.65
        7. if "attack" and already control lighthouse: 0.15
        8. if "attack" and don't gain control of lighthouse: 0
        9. if "connect" and not on a lighthouse: -1
        10. if "connect" and no new connection (connection not possible): 0
        11. if "connect" and connect three lighthouses: 1
        12. if "connect" and connect two lighthouses: 0.85 
        """
        state_lh = dict((tuple(lh["position"]), lh) for lh in state["lighthouses"])
        next_state_lh = dict((tuple(lh["position"]), lh) for lh in next_state["lighthouses"])

        # If status is False
        if status['success'] == False:
            return -1
        ### MOVE ###
        # If the move command is invalid
        elif action['command'] == "move":
            # If the move command is invalid
            if (state['position'][0] == next_state['position'][0]) and (state['position'][1] == next_state['position'][1]):
                return -1
            #If move and land on a lighthouse not owned by player
            elif next_state['position'] in state_lh.keys() and state_lh[next_state['position']]['owner'] != player.num:
                return 0.55
            # If move and increase bot's energy
            elif next_state['energy'] > state['energy']:
                return 0.35 
            else:
                return 0
        ### ATTACK ###
        elif action['command'] == "attack":
            # If attack and not on a lighthouse
            if state['position'] not in state_lh.keys():
                return -1 
            # If attack a lighthouse and gain control of it
            elif state_lh[state['position']]['owner'] != player.num and next_state_lh[next_state['position']]['owner'] == player.num:
                return 0.7 
            # If attack a lighthouse and already control it
            elif state_lh[state['position']]['owner'] == player.num :
                return 0.15 
            # If attack a lighthouse and not enough energy to gain control
            elif state_lh[state['position']]['owner'] != player.num and next_state_lh[next_state['position']]['owner'] != player.num:
                return 0
        ### CONNECT ###
        elif action['command'] == "connect":
            # If try to connect and not on a lighthouse
            if state['position'] not in state_lh.keys():
                return -1 
            # If try to connect and connection not possible (ex. doesn't own lighthouse, doesn't have key to any other lighthouse, etc.)
            elif (state_lh[state['position']]['owner'] != player.num or 
                  state_lh[state['position']]["connections"] == next_state_lh[next_state['position']]["connections"]): 
                return -0.85
            # If connect lighthouses
            elif (state_lh[state['position']]['owner'] == player.num and 
                  len(state_lh[state['position']]["connections"]) < len(next_state_lh[next_state['position']]["connections"])):
                # If connect three lighthouses
                new_connection = list(set(next_state_lh[next_state['position']]["connections"])-set(state_lh[state['position']]["connections"]))[0]
                if any(i in next_state_lh[next_state['position']]["connections"] for i in next_state_lh[new_connection]["connections"]):
                    return 1
                # If connect two lighthouses
                else:
                    return 0.85       
        elif action['command'] == "pass":
            return -1
        else:
            return 0
    

    def train(self, max_updates=1000000000000, num_steps_update=256):
        game_view = [view.GameView(self.game[i]) for i in range(len(self.game))]
        update = 0
        round = 0
        running = True
        
        while update < max_updates and running: 
            ###################################
            # Get experiences to update agent #
            ###################################
            for i in range(num_steps_update):
                # Event handler for game engine
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                for i in range(len(self.game)):
                    self.game[i].pre_round()
                    game_view[i].update()

                player_idx = 0
                for bot in self.bots:
                    player = [self.game[i].players[player_idx] for i in range(len(self.game))]

                    ####################################################
                    # If round 0, Get initial state and initialize bot
                    ####################################################
                    if round == 0:
                        state = []
                        bot.player_num = player[0].num
                        bot.map = [self.game[i].island.map for i in range(len(self.game))]
                        state = [self.get_state(player[i], i) for i in range(len(self.game))]
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
                    status = [self.turn(player[i], action[i]) for i in range(len(self.game))]

                    if self.debug:
                        try:
                            bot.error(status["message"], action)
                        except:
                            pass
                    
                    for i in range(len(self.game)):
                        scores_temp = []
                        scores_temp.append(player[i].score)
                        game_view[i].update()
                    
                    bot.scores.append(scores_temp)
                    next_state = [self.get_state(player[i], i) for i in range(len(self.game))]
                    reward = [self.estimate_reward(action[i], state[i], next_state[i], player[i], status[i]) for i in range(len(self.game))]
                    transition = [state, action, reward, next_state]
                    bot.transitions.append(transition)
                    bot.transitions_temp.append(transition)

                    player_idx += 1
                
                for i in range(len(self.game)):
                    self.game[i].post_round()

                ###########################################
                # Print the scores after each round
                ###########################################

                s = "########### ROUND %d SCORE: " % round
                for i in range(len(self.bots)):
                    s += "P%d: %d " % (i, self.game[i].players[i].score)
                print(s)

                round += 1
            
            update += 1
            print("update: ", update)
                
            ###########################################
            # Optimize models
            ###########################################
            for bot in self.bots:
                bot.optimize_model(bot.transitions_temp)
                bot.transitions_temp = []
                print("updated policy")
            
                policy_loss = pd.DataFrame()
                policy_loss[str(bot.player_num)] = bot.policy_loss_list
                os.makedirs('./losses', exist_ok=True)
                policy_loss.to_csv(f'./losses/{str(bot.player_num)}_policy_loss.csv', index_label='episode')
            
                bot.save_trained_model()
    
    def run(self, max_rounds=None):
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
                    bot.player_num = player[0].num
                    bot.map = [self.game[i].island.map for i in range(len(self.game))]
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
                reward = self.estimate_reward(action, state, next_state, player, status)
                transition = [state, action, reward, next_state]
                bot.transitions.append(transition)
                bot.transitions_temp.append(transition)

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
            

                
     