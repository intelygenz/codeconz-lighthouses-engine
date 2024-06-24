package engine

import "time"

/*

{ "game":
	{
		"gameId": 1,
		"rounds": [
			{
				"roundId": 1,
				"turns": [
					{
						"turnId": 1,
						"energyMap": [
							[-1, -1, 25, 15, -1],
							[-1, -1, 25, 15, -1],
							[-1, -1, 25, 15, -1],
							[-1, -1, 25, 15, -1]
						],
						"lighthouses": [
							{
								"lighthouseId": 1,
								"energy": 10,
								"position": { "x": 0, "y": 0 },
								"owner": 1,
								"connections": [2, 3]
							}
						],
						"players": [
							{
								"playerId": 1,
								"position": { "x": 0, "y": 0 },
								"energy": 10,
								"keys": [1, 2]
							}
						],
						"score": [
							{
								"playerId": 1,
								"score": 10
							}
						]
					}
				]
			}
		]
	}
}
*/

type State struct {
	turns []TurnState
}

type TurnState struct {
	ts    time.Time
	state []byte
}

func NewState(turns, players int) *State {
	return &State{
		turns: make([]TurnState, turns*players),
	}
}

func NewTurnState(ts time.Time, state []byte) *TurnState {
	return &TurnState{
		ts:    ts,
		state: state,
	}
}
