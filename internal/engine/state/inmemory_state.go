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

func (m *InMemoryState) SetNewRound(rId int, gs GameStatus) {
	r := Round{
		Id:    rId,
		Setup: gs,
	}
	m.GameState.Rounds = append(m.GameState.Rounds, r)
}
func (m *InMemoryState) AddPlayerTurn(rId int, ts Turn) {
	for _, r := range m.GameState.Rounds {
		if r.Id == rId {
			r.Turns = append(r.Turns, ts)
		}
	}
}
func (m *InMemoryState) DumpToFileFinalStateInJson() error {
	data, _ := json.Marshal(m.GameState)

	fName := fmt.Sprintf("./output/game-%s.json", time.Now().Format("2006_01_02_15_04_05"))
	if err := os.WriteFile(fName, data, 0644); err != nil {
		return err
	}

	return nil
}
