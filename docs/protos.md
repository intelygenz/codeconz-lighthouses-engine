# Protocol Documentation
<a name="top"></a>

## Table of Contents

- [proto/game.proto](#proto_game-proto)
    - [Lighthouse](#-Lighthouse)
    - [MapRow](#-MapRow)
    - [NewAction](#-NewAction)
    - [NewPlayer](#-NewPlayer)
    - [NewPlayerInitialState](#-NewPlayerInitialState)
    - [NewTurn](#-NewTurn)
    - [PlayerID](#-PlayerID)
    - [PlayerReady](#-PlayerReady)
    - [Position](#-Position)
  
    - [Action](#-Action)
  
    - [GameService](#-GameService)
  
- [Scalar Value Types](#scalar-value-types)



<a name="proto_game-proto"></a>
<p align="right"><a href="#top">Top</a></p>

## proto/game.proto
Game related messages.

This file contains the messages that are used to communicate between the game server and the players.


<a name="-Lighthouse"></a>

### Lighthouse
Represents a lighthouse in the game.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| Position | [Position](#Position) |  | Position of the lighthouse |
| Owner | [int32](#int32) |  | Owner of the lighthouse |
| Energy | [int32](#int32) |  | Energy of the lighthouse |
| Connections | [Position](#Position) | repeated | Connections of the lighthouse |
| HaveKey | [bool](#bool) |  | Have key to the lighthouse |






<a name="-MapRow"></a>

### MapRow
Represents the game map as a list of rows.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| Row | [int32](#int32) | repeated | Row of the map |






<a name="-NewAction"></a>

### NewAction
Represents the action that a player took in a turn.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| Action | [Action](#Action) |  | Action: 0 = pass, 1 = move, 2 = attack, 3 = connect |
| Destination | [Position](#Position) |  | Destination position for the action |
| Energy | [int32](#int32) |  | Energy used for the action |






<a name="-NewPlayer"></a>

### NewPlayer
Represents a response for a new player that joined the game.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| name | [string](#string) |  | Name of the player |
| serverAddress | [string](#string) |  | Address of the server |






<a name="-NewPlayerInitialState"></a>

### NewPlayerInitialState
Represents the initial state of the game for a given player.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| PlayerID | [int32](#int32) |  | Player ID |
| PlayerCount | [int32](#int32) |  | Number of players in the game |
| Position | [Position](#Position) |  | Initial position of the player |
| Map | [MapRow](#MapRow) | repeated | Complete map of the game |
| Lighthouses | [Lighthouse](#Lighthouse) | repeated | Lighthouses in the game |






<a name="-NewTurn"></a>

### NewTurn
Represents a new turn in the game for a given player.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| Position | [Position](#Position) |  | Current position of the player |
| Score | [int32](#int32) |  | Current score of the player |
| Energy | [int32](#int32) |  | Current energy of the player |
| View | [MapRow](#MapRow) | repeated | Current view of the player surroundings |
| Lighthouses | [Lighthouse](#Lighthouse) | repeated | Current state of the lighthouses |






<a name="-PlayerID"></a>

### PlayerID
Represents a player ID.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| PlayerID | [int32](#int32) |  | Player ID |






<a name="-PlayerReady"></a>

### PlayerReady
Represents the player ready state.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| Ready | [bool](#bool) |  | Ready state: 0 = not ready, 1 = ready |






<a name="-Position"></a>

### Position
Represents a position in the game map.


| Field | Type | Label | Description |
| ----- | ---- | ----- | ----------- |
| X | [int32](#int32) |  | X coordinate |
| Y | [int32](#int32) |  | Y coordinate |





 


<a name="-Action"></a>

### Action
Represents the actions that a player can take in a turn.

| Name | Number | Description |
| ---- | ------ | ----------- |
| PASS | 0 | Do nothing |
| MOVE | 1 | Move to a new position |
| ATTACK | 2 | Attack a lighthouse |
| CONNECT | 3 | Connect to a lighthouse |


 

 


<a name="-GameService"></a>

### GameService
Represents the game services.

| Method Name | Request Type | Response Type | Description |
| ----------- | ------------ | ------------- | ------------|
| Join | [.NewPlayer](#NewPlayer) | [.PlayerID](#PlayerID) |  |
| InitialState | [.NewPlayerInitialState](#NewPlayerInitialState) | [.PlayerReady](#PlayerReady) |  |
| Turn | [.NewTurn](#NewTurn) | [.NewAction](#NewAction) |  |

 



## Scalar Value Types

| .proto Type | Notes | C++ | Java | Python | Go | C# | PHP | Ruby |
| ----------- | ----- | --- | ---- | ------ | -- | -- | --- | ---- |
| <a name="double" /> double |  | double | double | float | float64 | double | float | Float |
| <a name="float" /> float |  | float | float | float | float32 | float | float | Float |
| <a name="int32" /> int32 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint32 instead. | int32 | int | int | int32 | int | integer | Bignum or Fixnum (as required) |
| <a name="int64" /> int64 | Uses variable-length encoding. Inefficient for encoding negative numbers – if your field is likely to have negative values, use sint64 instead. | int64 | long | int/long | int64 | long | integer/string | Bignum |
| <a name="uint32" /> uint32 | Uses variable-length encoding. | uint32 | int | int/long | uint32 | uint | integer | Bignum or Fixnum (as required) |
| <a name="uint64" /> uint64 | Uses variable-length encoding. | uint64 | long | int/long | uint64 | ulong | integer/string | Bignum or Fixnum (as required) |
| <a name="sint32" /> sint32 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int32s. | int32 | int | int | int32 | int | integer | Bignum or Fixnum (as required) |
| <a name="sint64" /> sint64 | Uses variable-length encoding. Signed int value. These more efficiently encode negative numbers than regular int64s. | int64 | long | int/long | int64 | long | integer/string | Bignum |
| <a name="fixed32" /> fixed32 | Always four bytes. More efficient than uint32 if values are often greater than 2^28. | uint32 | int | int | uint32 | uint | integer | Bignum or Fixnum (as required) |
| <a name="fixed64" /> fixed64 | Always eight bytes. More efficient than uint64 if values are often greater than 2^56. | uint64 | long | int/long | uint64 | ulong | integer/string | Bignum |
| <a name="sfixed32" /> sfixed32 | Always four bytes. | int32 | int | int | int32 | int | integer | Bignum or Fixnum (as required) |
| <a name="sfixed64" /> sfixed64 | Always eight bytes. | int64 | long | int/long | int64 | long | integer/string | Bignum |
| <a name="bool" /> bool |  | bool | boolean | boolean | bool | bool | boolean | TrueClass/FalseClass |
| <a name="string" /> string | A string must always contain UTF-8 encoded or 7-bit ASCII text. | string | String | str/unicode | string | string | string | String (UTF-8) |
| <a name="bytes" /> bytes | May contain any arbitrary sequence of bytes. | string | ByteString | str | []byte | ByteString | string | String (ASCII-8BIT) |

