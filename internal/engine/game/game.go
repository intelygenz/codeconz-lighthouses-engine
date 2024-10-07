package game

import (
	"fmt"
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/state"
)

type GameI interface {
	AddNewPlayer(np *player.Player) error
	GetPlayers() []*player.Player
	GetPlayerByID(id int) *player.Player
	CreateInitialState(p *player.Player) *PlayerInitialState
	SendInitialState()
	CalcPlayersScores()
	StartGame()
}

type Game struct {
	gameStartAt time.Time
	players     []*player.Player
	gameMap     board.BoardI
	turns       int
	state       state.State
}

func NewGame(islandPath string, turns int) GameI {
	gState := state.NewInMemoryState()
	return &Game{
		gameStartAt: time.Now(),
		players:     []*player.Player{},
		gameMap:     board.NewBoard(islandPath),
		turns:       turns,
		state:       gState,
	}
}

func (e *Game) AddNewPlayer(np *player.Player) error {
	np.ID = len(e.players) + 1
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
		var lines []Line
		for _, l := range e.gameMap.GetLightHouses() {
			if l.Owner == p.ID {
				// SCORE: sum 2 points for each lighthouse the player owns
				p.Score += 2

				// Calculate pairs of connected lighthouses
				for _, conn := range l.Connections {
					l := Line{A: &l.Position, B: &conn.Position}
					lines = append(lines, l)
				}
			}
		}
		// NOTE: as we are looping players and lighthouses,
		// lines will be duplicated for each pair of lighthouses connected
		// ex: if lighthouse 1 and 4 are connected, lines are: [[1, 4], [4,1]]
		// but it is only 1 connection between lighthouses 1 and 4
		connections := int(len(lines) / 2)

		// SCORE: sum 2 points for each pair of lighthouses connected
		p.Score += connections * 2

		// SCORE: get triangles for the player and calculate score inside each triangle
		triangles := GenerateTrianglesFromLines(lines)
		for _, t := range triangles {
			for _, coord := range renderTriangle(t) {
				if e.gameMap.IsIsland(coord) {
					fmt.Printf("one more triangle to sum!\n")
					p.Score++
				}
			}
		}
	}
}
