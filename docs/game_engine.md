# Game Engine
The engine is responsible for managing the game state and the game rules.

It will run three main services to manage the game:
- `Join` service: This service will run for a given time window defined at the `game.join_timeout` configuration. It will allow the players to join the game.
- `InitialState` service: This service will send the initial state of the game to the players.
- `Turn` service: Each turn the server will call this service to each player to get the player's action.

> Checkout protos documentation at [protos docs](protos.md).

> Checkout the [project README](../README.md) for instructions on how to execute the Game Engine, and the Sample bots.

> To create and test on new maps, you can use the [Map Generator](https://dovixman.github.io/lighthouses_map_generator/)

## Loading game configuration
The game engine will load the configuration from the environment variables or the configuration file.

- `listen_address` (defaults to `50051`)
- `join_timeout` (defaults to `5 seconds`)
- `turn_request_timeout` (defaults to `100 milliseconds`)
- `turns` (defaults to `15`)
- `board_path` (defaults to `./maps/island_simple.txt`)
- `verbosity` (defaults to `true`)
- `time_between_rounds` (defaults to `1 second`)

> You can check the default values at the [Configuration file](../cfg.yaml)

## Joining the game
The players bot will call the Join service on the game server.

The player will send a [NewPlayer](protos.md#newplayer) request with the following fields:
- `Name`: The name of the player.
- `ServerAddress`: The address on where to reach the player.

The game server will process the request and will send the response [PlayerID](protos.md#playerid) with the ID of the player:
- `PlayerID`: The ID of the player.

## Each player receives the initial state of the game
Once all players joined the game, the game server will send the initial state of the game to each player.

The game server will send a [NewPlayerInitialState](protos.md#newplayerinitialstate) response with the following fields:
- `PlayerID`: The ID of the player.
- `PlayerCount`: The number of players in the game.
- `Position`: The initial position of the player.
- `Map`: The complete map of the game.
- `Lighthouses`: The lighthouses in the game.

Once received and processed, the player will send a [PlayerReady](protos.md#playerready) request with the following fields:
- `Ready`: A boolean indicating if the player is ready to start the game.

## The game starts
The game will execute the number of turns defined in the `game.turns` configuration.

For each turn, the game will do the next actions in the following order:

### Calculate energy
- Calculate each Island cell **energy** based on the distance from the Lighthouses using the next formula (`energy += floor(5 - distance_to_lighthouse)`) with a maximum value of `100`.
- Calculate each Lighthouse **energy** (each turn each Lighthouse looses `10` energy points).
- Calculate each Player **energy** by extracting current positions cell energy
  - If there is only one player on the cell, **all the energy will be absorbed by the player**
  - If there are more than one player on the cell, **the energy will be equally distributed between the players**

### Ask each player for their turn
The game server will request each player's action by invoking the Turn service.

Each player's turn will be processed in the order they joined the game, based on their ID.

The player will receive a [NewTurn](protos.md#newturn) request with the following fields:
  - `Position`: The current position of the player.
  - `Score`: The current score of the player.
  - `Energy`: The current energy of the player.
  - `View`: The current view of the player surroundings.
  - `Lighthouses`: The current state of the lighthouses.

The player will process the request and will send a [PlayerTurn](protos.md#newaction) response back to the game server with the following fields:
  - `Action`: The action to perform on this turn.
  - `Destination`: The destination position for the action.
  - `Energy`: The energy used for the action.

The game engine will process and apply the player's action.

### Calculate player scores
When all the turns are executed, the player scores are calculated based on the number of connections and triangles they have.

### Game ends
When all the turns are executed, the game will write the game status to a file containing the game results for each player.
