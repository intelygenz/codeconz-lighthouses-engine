#!/usr/bin/python
# -*- coding: utf-8 -*-
import engine, view
from collections import deque
import numpy as np

# ==============================================================================
# ROBOT
# Los robots definidos deben heredar de esta clase.
# ==============================================================================

class Bot(object):
    """Bot base. Este bot no hace nada (pasa todos los turnos)."""
    NAME = "NullBot"

    # ==========================================================================
    # Comportamiento del bot
    # Métodos a implementar / sobreescribir (opcionalmente)
    # ==========================================================================

    def __init__(self):
        """Inicializar el bot: llamado al comienzo del juego."""
        pass

    def play(self, state):
        """Jugar: llamado cada turno.
        Debe devolver una acción (jugada).
        
        state: estado actual del juego.
        """
        return self.nop()

    def success(self):
        """Éxito: llamado cuando la jugada previa es válida."""
        pass

    def error(self, message, last_move):
        """Error: llamado cuando la jugada previa no es válida."""
        print("Recibido error: %s", message)
        print("Jugada previa: %r", last_move)

    # ==========================================================================
    # Jugadas posibles
    # No es necesario sobreescribir estos métodos.
    # ==========================================================================

    def nop(self):
        """Pasar el turno"""
        return {
            "command": "pass",
        }

    def move(self, x, y):
        """Mover a una casilla adyacente
        x: delta x (0, -1, 1)
        y: delta y (0, -1, 1)
        """
        return {
            "command": "move",
            "x": x,
            "y": y
        }

    def attack(self, energy):
        """Atacar a un faro
        energy: energía (entero positivo)
        """
        return {
            "command": "attack",
            "energy": energy
        }

    def connect(self, destination):
        """Conectar a un faro remoto
        destination: tupla o lista (x,y): coordenadas del faro remoto
        """
        return {
            "command": "connect",
            "destination": destination
        }


# ==============================================================================
# Interfaz
# ==============================================================================

class CommError(Exception):
    pass


class Interface(object):
    def __init__(self, game, bots):
        self.game = game
        self.bots = bots

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

    def run(self):
        game_view = view.GameView(self.game)

        # Send initial state to every bot
        player_idx = 0
        for bot in self.bots:
            # Every bot receives initial state
            self.bots[player_idx].player_num = self.game.players[player_idx].num
            self.bots[player_idx].player_count = len(self.game.players)
            self.bots[player_idx].position = self.game.players[player_idx].pos
            self.bots[player_idx].map = self.game.island.map
            self.bots[player_idx].lighthouses = map(tuple, 
                                                    list(self.game.lighthouses.keys()))

            if self.bots[player_idx].NAME[0] == 'REINFORCE':
                player = self.game.players[player_idx]
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

                state =  {
                    "position": player.pos,
                    "score": player.score,
                    "energy": player.energy,
                    "view": self.game.island.get_view(player.pos),
                    "lighthouses": lighthouses
                }

                bot.initialize_game(state)

            player_idx += 1

        round = 0
        while True:
            self.game.pre_round()
            game_view.update()
            
            player_idx = 0
            for bot in self.bots:
                player = self.game.players[player_idx]

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

                state =  {
                    "position": player.pos,
                    "score": player.score,
                    "energy": player.energy,
                    "view": self.game.island.get_view(player.pos),
                    "lighthouses": lighthouses
                }

                move = bot.play(state)

                s = "########### ROUND %d ENERGY: " % round
                s += "P%d: %s " % (player_idx, player.energy)
                # print(s)

                status = self.turn(player, move)

                if status["success"]:
                    bot.success()
                else:
                    bot.error(status["message"], move)

                game_view.update()
                player_idx += 1

            self.game.post_round()
            
            # Print the scores after each round
            s = "########### ROUND %d SCORE: " % round
            for i in range(len(self.bots)):
                s += "P%d: %d " % (i, self.game.players[i].score)
            # print(s)
            
            round += 1

    def train_reinforce(self, n_training_episodes=10, max_t=10, print_every=1, save_model=True, use_saved_model=False):
        self.max_t = max_t
        scores_deque = deque(maxlen=100)
        scores = []

        for i_episode in range(1, n_training_episodes+1):
            game_view = view.GameView(self.game)

            # Send initial state to every bot
            player_idx = 0
            for bot in self.bots:
                # Every bot receives initial state
                self.bots[player_idx].player_num = self.game.players[player_idx].num
                self.bots[player_idx].player_count = len(self.game.players)
                self.bots[player_idx].position = self.game.players[player_idx].pos
                self.bots[player_idx].map = self.game.island.map
                self.bots[player_idx].lighthouses = map(tuple, 
                                                        list(self.game.lighthouses.keys()))

                if self.bots[player_idx].NAME[0] == 'REINFORCE':
                    train_idx = player_idx
                    player = self.game.players[player_idx]
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

                    state =  {
                        "position": player.pos,
                        "score": player.score,
                        "energy": player.energy,
                        "view": self.game.island.get_view(player.pos),
                        "lighthouses": lighthouses
                    }

                    bot.initialize_game(state)

                player_idx += 1

            round = 0
            self.saved_log_probs = []
            self.rewards = []
            for t in range(max_t):
                self.game.pre_round()
                game_view.update()
                
                player_idx = 0
                for bot in self.bots:
                    player = self.game.players[player_idx]

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

                    state =  {
                        "position": player.pos,
                        "score": player.score,
                        "energy": player.energy,
                        "view": self.game.island.get_view(player.pos),
                        "lighthouses": lighthouses
                    }
                    if player_idx == train_idx:
                        move, log_prob = bot.play_train(state)
                        self.saved_log_probs.append(log_prob)
                    else:
                        move = bot.play(state)

                    status = self.turn(player, move)

                    if status["success"]:
                        bot.success()
                    else:
                        bot.error(status["message"], move)

                    game_view.update()
                    player_idx += 1

                self.game.post_round()
                
                # Print the scores after each round
                #reward = self.game.players[train_idx].score
                reward = self.game.players[train_idx].energy
                self.rewards.append(reward)    
                round += 1
            scores_deque.append(sum(self.rewards))
            scores.append(sum(self.rewards))
            if bot.NAME == 'REINFORCE':
                bot.update_policy(self.rewards, max_t, self.saved_log_probs)

            if i_episode % print_every == 0:
                print('Episode{}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
        if save_model:
            pass
            # bot.save_model()

        print(scores)