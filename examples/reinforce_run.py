import time
import argparse
import grpc
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, os
import numpy as np
import random
from concurrent import futures
from grpc import RpcError
from google.protobuf import json_format
from torch.nn.modules import Module
from torch.distributions import Categorical
from sklearn.preprocessing import StandardScaler
from collections import deque

from internal.handler.coms import game_pb2
from internal.handler.coms import game_pb2_grpc as game_grpc


timeout_to_response = 1  # 1 second

SAVED_MODEL = True
STATE_MAPS = True
MODEL_PATH = './examples/saved_model'
MODEL_FILENAME = 'reinforce_test_cnn_transfer.pth'
ACTIONS = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1), "attack", "connect", "pass")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BotGameTurn:
    def __init__(self, turn, action):
        self.turn = turn
        self.action = action

# Define the neural network for the policy
class PolicyMLP(nn.Module):
    def __init__(self, s_size, a_size, layers_data: list):
        super(PolicyMLP, self).__init__()
        self.layers = nn.ModuleList()
        input_size = s_size
        for size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, size))
            input_size = size
            if activation is not None:
                assert isinstance(activation, Module), \
                    "Each tuple should contain a layer size (int) and an activation (ex. nn.ReLU())."
                self.layers.append(activation)
        self.layers.append(nn.Linear(size, a_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.log_softmax(x, dim=-1)
    
    def act(self, state):
        """
        Given a state, take action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

class PolicyCNN(nn.Module):
    def __init__(self, num_maps, a_size: list):
        super(PolicyCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=num_maps, out_channels=32, kernel_size=5), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*12*12, 512),
            nn.ReLU(),
            nn.Linear(512, a_size)
        )

    def forward(self, x):
        x = self.network(x)
        return torch.log_softmax(x, dim=-1)
    
    def act(self, state):
        """
        Given a state, take action
        """
        state = np.transpose(state, (2,0,1))
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)
    

class BotGame:
    def __init__(self, player_num=None):
        self.player_num = player_num
        self.initial_state = None
        self.turn_states = []
        self.countT = 1
        self.state_maps = STATE_MAPS # use maps for state (CNN): True, or array for state (MLP): False
        self.a_size = len(ACTIONS)
        self.layers_data = [(64, nn.ReLU()), (64, nn.ReLU())]
        self.model_path = MODEL_PATH
        self.model_filename = MODEL_FILENAME
        self.use_saved_model = SAVED_MODEL
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    def load_saved_model(self):
        if os.path.isfile(os.path.join(self.model_path, self.model_filename)):
            self.policy.load_state_dict(torch.load(os.path.join(self.model_path, self.model_filename)))
            print(f"Loaded saved model: {self.model_filename}")
        else:
            print("No saved model")

    def initialize_game(self, turn):
        self.saved_log_probs = []
        if self.state_maps:
            print("Using maps for state: PolicyCNN")
            state = self.convert_state_cnn(turn)
            self.num_maps = state.shape[2]
            self.policy = PolicyCNN(self.num_maps, self.a_size).to(self.device)
        else:
            print("Using array for state: PolicyMLP")
            state = self.convert_state_mlp(turn)
            self.s_size = len(state)
            self.policy = PolicyMLP(self.s_size, self.a_size, self.layers_data).to(self.device)
        if self.use_saved_model:
            self.load_saved_model()

    def convert_state_mlp(self, turn):
        # Create array for view data
        view = []
        for i in range(len(turn.View)):
            view = view + list(turn.View[i].Row)
        cx = turn.Position.X
        cy = turn.Position.Y
        cx_min, cx_max = cx-3, cx+3
        cy_min, cy_max = cy-3, cy+3
        lighthouses = np.zeros((7,7), dtype=int)
        lighthouses_dict = dict((tuple((lh.Position.X, lh.Position.Y)), lh.Owner) for lh in turn.Lighthouses)
        for key in lighthouses_dict.keys():
            if cx_min <= key[0] <= cx_max and cy_min <= key[1] <= cy_max:
                lighthouses[key[0]+3-cx, key[1]+3-cy] = lighthouses_dict[key] # for owner
               # lighthouses[key[0]+3-cx, key[1]+3-cy] = lighthouses_dict[key] + 1 for energy
        lighthouses_info = []
        # Create array for lighthouses data (within 3 steps of the bot)
        for i in range(len(lighthouses)):
            lighthouses_info = lighthouses_info + list(lighthouses[i])
        new_state = np.array([turn.Position.X, turn.Position.Y, turn.Energy, len(turn.Lighthouses)] + view + lighthouses_info)
        sc = StandardScaler()
        new_state = sc.fit_transform(new_state.reshape(-1, 1))
        new_state = new_state.squeeze()
        return new_state
    
    def z_score_scaling(self, arr):
        arr_mean = np.mean(arr)
        arr_std = np.std(arr)
        scaled_arr = (arr - arr_mean) / arr_std
        return scaled_arr

    def convert_state_cnn(self, turn):
        # Create base layer that will serve as the base for all layers of the state
        # This layer has zeros in all cells except the border cells in which the value is -1
        map = []
        for i in range(len(self.initial_state.Map)):
            map.append(self.initial_state.Map[i].Row)
        base_layer = np.array(map.copy())

        # Create player layer which has the value of the energy of the player + 1 where the player is located
        # 1 is added to the energy to cover the case that the energy of the player is 0
        player_layer = base_layer.copy()
        x, y = turn.Position.X, turn.Position.Y
        player_layer[x,y] = 1 + turn.Energy
        player_layer = self.z_score_scaling(player_layer)
    
        # Create view layer with energy level near the player
        view_layer = base_layer.copy()
        view = []
        for i in range(len(turn.View)):
            view.append(turn.View[i].Row)
        view = np.array(view)
        start_row, start_col = x-3, y-3
        if y+3 > view_layer.shape[1]-1:
            adjust = view_layer.shape[1]-1 - (y+3)
            view = view[:,:adjust]
        if x+3 > view_layer.shape[0]-1:
            adjust = view_layer.shape[0]-1 - (x+3)
            view = view[:adjust,:]
        if y-3 < 0:
            adjust = 3-y
            view = view[:,adjust:]
            start_col = 0
        if x-3 < 0:
            adjust = 3-x
            view = view[adjust:,:]
            start_row = 0
        view_layer[start_row:start_row+view.shape[0], start_col:start_col+view.shape[1]] = view
        view_layer = self.z_score_scaling(view_layer)

        # Create layer that has the energy of the lighthouse + 1 where the lighthouse is located
        # 1 is added to the lighthouse energy to cover the case that the energy of the lighthouse is 0
        # lh_energy_layer = base_layer.copy()
        # lh = turn.Lighthouses
        # for i in range(len(lh)):
        #     x, y = lh[i].Position.X, lh[i].Position.Y
        #     lh_energy_layer[x,y] = 1 - lh.Energy
        # lh_energy_layer = self.z_score_scaling(lh_energy_layer)

        # Create layer that has the number of the player that controls each lighthouse
        # If no player controls the lighthouse, then a value of -1 is assigned
        lh_control_layer = base_layer.copy()
        lh = turn.Lighthouses
        for i in range(len(lh)):
            x, y = lh[i].Position.X, lh[i].Position.Y
            lh_control_layer[x,y] = lh[i].Owner
        lh_control_layer = self.z_score_scaling(lh_control_layer)

        # Create layer that indicates the lighthouses that are connected
        # If the lighthouse is not connected, then a value of -1 is assigned, if it is connected then it is 
        # assigned the number of connections that it has
        lh_connections_layer = base_layer.copy()
        for i in range(len(lh)):
            x, y = lh[i].Position.X, lh[i].Position.Y
            if lh[i].Connections:
                lh_connections_layer[x,y] = len(lh[i].Connections)
            else:
                lh_connections_layer[x,y] = -1
        lh_connections_layer = self.z_score_scaling(lh_connections_layer)

        # Create layer that indicates if the player has the key to the light house
        # Assign value of 1 if has key and -1 if does not have key
        lh_key_layer = base_layer.copy()
        for i in range(len(lh)):
            x, y = lh[i].Position.X, lh[i].Position.Y
            if lh[i].HaveKey:
                lh_key_layer[x,y] = 1
            else:
                lh_key_layer[x,y] = -1
        lh_key_layer = self.z_score_scaling(lh_key_layer)

        # Concatenate the maps into one state
        player_layer = np.expand_dims(player_layer, axis=2)
        view_layer = np.expand_dims(view_layer, axis=2)
       # lh_energy_layer = np.expand_dims(lh_energy_layer, axis=2)
        lh_control_layer = np.expand_dims(lh_control_layer, axis=2)
        lh_connections_layer = np.expand_dims(lh_connections_layer, axis=2)
        lh_key_layer = np.expand_dims(lh_key_layer, axis=2)

        new_state = np.concatenate((player_layer, view_layer, lh_control_layer, lh_connections_layer, lh_key_layer), axis=2)
        return new_state

    def valid_lighthouse_connections(self, turn):
        cx = turn.Position.X
        cy = turn.Position.Y
        lighthouses = dict((tuple((lh.Position.X, lh.Position.Y)), lh) for lh in turn.Lighthouses)
        possible_connections = []
        if (cx, cy) in lighthouses:
            if lighthouses[(cx, cy)].Owner == self.player_num:
                for dest in lighthouses:
                    if (dest != (cx, cy) and
                        lighthouses[dest].HaveKey and
                        [cx, cy] not in lighthouses[dest].Connections and
                        lighthouses[dest].Owner == self.player_num):
                        possible_connections.append(dest)
        return possible_connections

    def new_turn_action(self, turn: game_pb2.NewTurn) -> game_pb2.NewAction:
        if self.countT == 1:
            self.initialize_game(turn)
        if self.state_maps:
            print("Using maps for state: PolicyCNN")
            new_state = self.convert_state_cnn(turn)
        else:
            print("Using array for state: PolicyMLP")
            new_state = self.convert_state_mlp(turn)
        action, log_prob = self.policy.act(new_state)
        self.saved_log_probs.append(log_prob)
        if ACTIONS[action] != "attack" and ACTIONS[action] != "connect" and ACTIONS[action] != "pass":
            move = ACTIONS[action]
            action = game_pb2.NewAction(
                Action=game_pb2.MOVE,
                Destination=game_pb2.Position(
                    X=turn.Position.X + move[0],
                    Y=turn.Position.Y + move[1]
                )
            )
            bgt = BotGameTurn(turn, action)
            self.turn_states.append(bgt)

            self.countT += 1
            return action

        elif ACTIONS[action] == "pass":
            action = game_pb2.NewAction(
                Action=game_pb2.MOVE,
                Destination=game_pb2.Position(
                    X=turn.Position.X,
                    Y=turn.Position.Y
                )
            )
            bgt = BotGameTurn(turn, action)
            self.turn_states.append(bgt)

            self.countT += 1
            return action
        
        elif ACTIONS[action] == "attack":
            energy = turn.Energy
            action = game_pb2.NewAction(
                Action=game_pb2.ATTACK,
                Energy=energy,
                Destination=game_pb2.Position(
                    X=turn.Position.X,
                    Y=turn.Position.Y
                )
            )
            bgt = BotGameTurn(turn, action)
            self.turn_states.append(bgt)

            self.countT += 1
            return action

        elif ACTIONS[action] == "connect":
            possible_connections = self.valid_lighthouse_connections(turn)
            if not possible_connections:
                action = game_pb2.NewAction(
                    Action=game_pb2.PASS,
                    Destination=game_pb2.Position(
                        X=turn.Position.X,
                        Y=turn.Position.Y
                    )
                )
                bgt = BotGameTurn(turn, action)
                self.turn_states.append(bgt)

                self.countT += 1
                return action
            else: 
                possible_connection = random.choice(possible_connections)
                action = game_pb2.NewAction(
                    Action=game_pb2.CONNECT,
                    Destination=game_pb2.Position(X=possible_connection[0], Y=possible_connection[1])
                )
                bgt = BotGameTurn(turn, action)
                self.turn_states.append(bgt)

                self.countT += 1
                return action


class BotComs:
    def __init__(self, bot_name, my_address, game_server_address, verbose=True):
        self.bot_id = None
        self.bot_name = bot_name
        self.my_address = my_address
        self.game_server_address = game_server_address
        self.verbose = verbose

    def wait_to_join_game(self):
        channel = grpc.insecure_channel(self.game_server_address)
        client = game_grpc.GameServiceStub(channel)

        player = game_pb2.NewPlayer(name=self.bot_name, serverAddress=self.my_address)

        while True:
            try:
                player_id = client.Join(player, timeout=timeout_to_response)
                self.bot_id = player_id.PlayerID
                print(f"Joined game with ID {player_id.PlayerID}")
                if self.verbose:
                    print(json_format.MessageToJson(player_id))
                break
            except RpcError as e:
                print(f"Could not join game: {e.details()}")
                time.sleep(1)


    def start_listening(self):
        print("Starting to listen on", self.my_address)

        # configure gRPC server
        grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            interceptors=(ServerInterceptor(),)
        )

        # registry of the service
        cs = ClientServer(bot_id=self.bot_id, verbose=self.verbose)
        game_grpc.add_GameServiceServicer_to_server(cs, grpc_server)

        # server start
        grpc_server.add_insecure_port(self.my_address)
        grpc_server.start()

        try:
            grpc_server.wait_for_termination()  # wait until server finish
        except KeyboardInterrupt:
            grpc_server.stop(0)


class ServerInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        start_time = time.time_ns()
        method_name = handler_call_details.method

        # Invoke the actual RPC
        response = continuation(handler_call_details)

        # Log after the call
        duration = time.time_ns() - start_time
        print(f"Unary call: {method_name}, Duration: {duration:.2f} nanoseconds")
        return response


class ClientServer(game_grpc.GameServiceServicer):
    def __init__(self, bot_id, verbose=False):
        self.bg = BotGame(bot_id)
        self.verbose = verbose

    def Join(self, request, context):
        return None

    def InitialState(self, request, context):
        print("Receiving InitialState")
        if self.verbose:
            print(json_format.MessageToJson(request))
        self.bg.initial_state = request
        print("initial state: ", self.bg.initial_state)
        return game_pb2.PlayerReady(Ready=True)

    def Turn(self, request, context):
        print(f"Processing turn: {self.bg.countT}")
        if self.verbose:
            print(json_format.MessageToJson(request))
        action = self.bg.new_turn_action(request)
        return action


def ensure_params():
    parser = argparse.ArgumentParser(description="Bot configuration")
    parser.add_argument("--bn", type=str, default="random-bot", help="Bot name")
    parser.add_argument("--la", type=str, default="localhost:3001", help="Listen address")
    parser.add_argument("--gs", type=str, default="localhost:50051", help="Game server address")

    args = parser.parse_args()

    if not args.bn:
        raise ValueError("Bot name is required")
    if not args.la:
        raise ValueError("Listen address is required")
    if not args.gs:
        raise ValueError("Game server address is required")

    return args.bn, args.la, args.gs


def main():
    verbose = False
    bot_name, listen_address, game_server_address = ensure_params()

    bot = BotComs(
        bot_name=bot_name,
        my_address=listen_address,
        game_server_address=game_server_address,
        verbose=verbose,
    )

    bot.wait_to_join_game()
    bot.start_listening()


if __name__ == "__main__":
    main()
