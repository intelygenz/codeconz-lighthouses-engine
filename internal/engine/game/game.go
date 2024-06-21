package game

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"time"
)

type Game struct {
	gameStartAt time.Time
	players     []player.Player
	gameMap     board.Board
	turns       int
}

func NewGame(islandPath string, turns int) *Game {
	return &Game{
		gameStartAt: time.Now(),
		players:     []player.Player{},
		gameMap:     board.NewBoard(islandPath),
		turns:       turns,
	}
}

func (e *Game) AddNewPlayer(np player.Player) error {
	// TODO: Add ID and location for the Player
	e.players = append(e.players, np)
	return nil
}

func (e *Game) CreateInitialState(p player.Player) *coms.NewPlayerInitialState {
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
