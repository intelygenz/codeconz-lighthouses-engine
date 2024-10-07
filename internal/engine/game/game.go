package game

import (
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/state"
	"github.com/twpayne/go-geom"
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
			// REVIEW: ojo, estamos duplicando el score por cada lighthouse?
			if l.Owner == p.ID {
				p.Score += 2
				p.Score += 2 * len(l.Connections)

				// Calculate pairs of connected lighthouses
				for _, conn := range l.Connections {
					// get lines to calculate later triangles for the player
					l := Line{A: &l.Position, B: &conn.Position}
					lines = append(lines, l)
					if conn.Owner == p.ID {
						p.Score += 2
					}
				}
			}
		}
		// get triangles for the player
		triangles := GenerateTrianglesFromLines(lines)
		// calculate score inside each triangle
		for _, t := range triangles {

			v0 := []int{int(t.A.X()), int(t.A.Y())}
			v1 := []int{int(t.B.X()), int(t.B.Y())}
			v2 := []int{int(t.C.X()), int(t.C.Y())}

			for _, point := range renderTriangle(v0, v1, v2) {
				pos := geom.Coord{float64(point[0]), float64(point[1])}
				if e.gameMap.IsIsland(pos) {
					p.Score++
				}
			}
		}
	}
}
