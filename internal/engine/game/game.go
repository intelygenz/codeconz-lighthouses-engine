package game

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"time"
)

type GameI interface {
	AddNewPlayer(np player.Player) error
	CreateInitialState(p player.Player) *PlayerInitialState
	StartGame()
}

type Game struct {
	gameStartAt time.Time
	players     []*player.Player
	gameMap     board.BoardI
	turns       int
}

func NewGame(islandPath string, turns int) GameI {
	return &Game{
		gameStartAt: time.Now(),
		players:     []*player.Player{},
		gameMap:     board.NewBoard(islandPath),
		turns:       turns,
	}
}

func (e *Game) AddNewPlayer(np player.Player) error {
	np.ID = len(e.players)

	np.Position = e.gameMap.GetRandomIslandLocation()

	e.players = append(e.players, &np)
	return nil
}
