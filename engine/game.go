package engine

import (
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/coms"
)

const (
	timeoutPlayerToAnswer = time.Millisecond * 100
	gameTurns             = 10
)

type Game struct {
	gameStartAt time.Time
	players     []Player
}

func NewGame() *Game {
	return &Game{
		gameStartAt: time.Now(),
		players:     []Player{},
	}
}

func (e *Game) AddNewPlayer(np Player) error {
	e.players = append(e.players, np)
	return nil
}

func (e *Game) CreateInitialState(p Player) *coms.NewPlayerInitialState {
	// TODO: implement the logic to create the initial state of the game
	lighthouseExample := &coms.Lighthouse{Position: &coms.Position{X: 1, Y: 1}}
	npst := &coms.NewPlayerInitialState{
		PlayerNum:   0,
		PlayerCount: 2,
		Position:    &coms.Position{X: 1, Y: 2},
		Map:         []*coms.MapRow{{Row: []int32{0, 0, 0, 0, 0}}},
		Lighthouses: []*coms.Lighthouse{lighthouseExample},
	}
	return npst
}

func (e *Game) StartGame() {
	e.gameStartAt = time.Now()

	viewExample := &coms.MapRow{
		Row: []int32{0, 0, 0, 0, 0},
	}
	lighthouseExample := &coms.Lighthouse{
		Position: &coms.Position{
			X: 1,
			Y: 1,
		},
		Energy: 100,
	}

	newTurnExample := &coms.NewTurn{
		Position: &coms.Position{
			X: 1,
			Y: 1,
		},
		Score:       10,
		Energy:      100,
		View:        []*coms.MapRow{viewExample},
		Lighthouses: []*coms.Lighthouse{lighthouseExample},
	}

	for i := 0; i < gameTurns; i++ {
		// TODO: calc new turn state
		for _, p := range e.players {
			gc := &GameClient{}
			comPlayer := &coms.NewPlayer{
				Name:          p.Name,
				ServerAddress: p.ServerAddress,
			}
			// TODO: calc the new turn with the last state of the game
			na := gc.requestTurn(comPlayer, newTurnExample)
			_ = na
			// TODO: apply the action to change the state of the game
		}
		// TODO: calc players scores
	}
}
