package engine

import "time"

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
