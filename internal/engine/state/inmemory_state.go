package state

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

type InMemoryState struct {
	GameState GameState
}

func NewInMemoryState() State {
	return &InMemoryState{
		GameState: GameState{},
	}
}

func (m *InMemoryState) SetGameState(gs GameState) {
	m.GameState = gs
}

func (m *InMemoryState) SetFinalGameState(gStatus *GameStatus) {
	m.GameState.FinalStatus = gStatus
}

func (m *InMemoryState) SetNewRound(rId int, gStatus *GameStatus) {
	r := Round{
		Id:    rId,
		Setup: gStatus,
		Turns: []*Turn{},
	}
	m.GameState.Rounds = append(m.GameState.Rounds, r)
}
func (m *InMemoryState) AddPlayerTurn(rId int, ts *Turn) {
	m.GameState.Rounds[rId-1].Turns = append(m.GameState.Rounds[rId-1].Turns, ts)
}

func (m *InMemoryState) DumpToFileFinalStateInJson() error {
	data, _ := json.Marshal(m.GameState)

	fName := fmt.Sprintf("./output/game-%s.json", time.Now().Format("2006_01_02_15_04_05"))
	if err := os.WriteFile(fName, data, 0644); err != nil {
		return err
	}

	return nil
}
