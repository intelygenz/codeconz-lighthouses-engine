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
		for _, p := range e.players {
			gc := &GameClient{}
			comPlayer := &coms.NewPlayer{
				Name:          p.Name,
				ServerAddress: p.ServerAddress,
			}
			gc.requestTurn(comPlayer, newTurnExample)
		}
	}
}
