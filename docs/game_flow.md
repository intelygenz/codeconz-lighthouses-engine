# Game Flow
- Initialize configuration:
    - `game.listen_address` (defaults to `50051`)
    - `game.join_timeout` (defaults to `5 seconds`)
    - `game.turn_request_timeout` (defaults to `100 milliseconds`)
    - `game.turns` (defaults to `15`)
    - `game.board_path` (defaults to `./maps/island_simple.txt`)
    - `game.verbosity` (defaults to `true`)
    - `game.time_between_rounds` (defaults to `1 second`)
- Initializes a new Game board with the defined map on `game.board_path`, and with the defined turns in `game.turns`
- Starts a gRPC server to let the players join the game on `game.listen_address` for the time defined in `game.join_timeout`
  - When the players join, they are given their `Player ID` back.
- Sends the initial state to all registered players containing
    - `PlayerID`
    - `PlayerCount`: Number of Players on the current game
    - `Position`: The initial position of the player
    - `Map`: The map representation as integers where
        - `0` is a Water cell, not playable
        - `1` is an Island cell, playable
    - `Lighthouses`: The lighthouses position
- Start the game
    - Set the initial game state with the Energy, the Player information and the Lighthouses information
    - For each turn in the turns defined in `game.turns` ...
        - Calculate each Island cell energy based on the distance from the Lighthouses using the next formula (`energy += floor(5 - distance_to_lighthouse)`) with a maximum value of `100`
        - Calculate each Lighthouse energy (each turn each Lighthouse looses `10` energy points)
        - Calculate each Player energy by extracting current positions cell energy
            - If there is only one player on the cell, all the energy will be absorbed by the player
            - If there are more than one player on the cell, the energy will be equally distributed between the players
        - Store new round information on the game state
        - Start a new round, for each player...
            - Request the new turn action to each player by sending information about the player
                - `Position`
                - `Score`
                - `Energy`
                - `View`: A map with the energy of the surroundings cells of the player up to 7 cells of distance
                    - `-1` is a Water cell, not playable
                    - `>0` is an Island or Lighthouse cells Energy
                - `Lighthouses` information, including
                    - `Position`
                    - `Energy`
                    - `Owner`: `PlayerID` owning the Lighthouse or `-1` if it has no owner
                    - `Connections`: The Lighthouse connections to other Lighthouses
                    - `HaveKey`: If the player has the key to the Lighthouse
            - Execute each player action, the actions can be
                - `Move` to given position
                - `Attack` a Lighthouse
                - `Connect` two Lighthouses
                - `Pass`, do nothing on this turn
            - Store the players turn in the game state
        - When all the runs are executed, the `player scores` are calc
        - 
        - 
        - clearulated based on the number of connections and triangles they have.
    - Write game status to a file