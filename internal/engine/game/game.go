package game

import (
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
)

type GameI interface {
	AddNewPlayer(np *player.Player) error
	GetPlayers() []*player.Player
	GetPlayerByID(id int) *player.Player
	CreateInitialState(p *player.Player) *PlayerInitialState
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

func (e *Game) AddNewPlayer(np *player.Player) error {
	np.ID = len(e.players)
	np.Position = e.gameMap.GetRandomPlayerInitialPosition()
	e.players = append(e.players, np)
	return nil
}

func (e *Game) GetPlayers() []*player.Player {
	return e.players
}

func (e *Game) GetPlayerByID(id int) *player.Player {
	for _, p := range e.players {
		if p.ID == id {
			return p
		}
	}
	return nil
}

func (e *Game) CalcPlayersScores() {
	for _, p := range e.players {
		for _, l := range e.gameMap.GetLightHouses() {
			if l.Owner == p.ID {
				p.Score += 2
				p.Score += 2 * len(l.Connections)
				// TODO: add 1 point for each cell inside connected triangles
			}
		}
	}
}
