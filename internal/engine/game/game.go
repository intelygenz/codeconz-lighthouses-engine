package game

import (
	"time"

	"github.com/twpayne/go-geom"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
)

type GameI interface {
	AddNewPlayer(np *player.Player) error
	GetPlayers() []*player.Player
	GetPlayerByID(id int) *player.Player
	CreateInitialState(p *player.Player) *PlayerInitialState
	SendInitialState()
	StartGame()
}

type Game struct {
	gameStartAt time.Time
	players     []*player.Player
	gameMap     board.BoardI
	turns       int
}

type Line struct {
	A, B *geom.Coord
}

type Triangle struct {
	A, B, C geom.Coord
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

// TODO pending
func (e *Game) CalcPlayersScores() {
	for _, p := range e.players {
		for _, l := range e.gameMap.GetLightHouses() {
			if l.Owner == p.ID {
				p.Score += 2
				p.Score += 2 * len(l.Connections)
				// TODO: add 1 point for each cell inside connected triangles

				// Calculate pairs of connected lighthouses
				for _, conn := range l.Connections {
					if conn.Owner == p.ID {
						p.Score += 2
					}
				}
			}
		}
	}
}

// NormalizeLine ensures that a line is always stored with the smaller point first
func (e *Game) NormalizeLine(line Line) Line {
	if (line.A.X() < line.B.X()) || (line.A.X() == line.B.X() && line.A.Y() < line.B.Y()) {
		return line
	}

	return Line{A: line.B, B: line.A}
}

// TODO: GenerateTrianglesFromLines generates all possible triangles from a given array of lines
func (e *Game) GenerateTrianglesFromLines(lines []Line) []Triangle {
	var triangles []Triangle

	// Normalize lines to handle duplicates with reversed points
	linesMap := make(map[Line]bool)
	for _, line := range lines {
		normalizedLine := e.NormalizeLine(line)
		linesMap[normalizedLine] = true
	}

	// Create a slice of unique lines
	var uniqueLines []Line
	for line := range linesMap {
		uniqueLines = append(uniqueLines, line)
	}

	// Generate all possible triangles from unique lines
	n := len(uniqueLines)
	for i := 0; i < n-2; i++ {
		for j := i + 1; j < n-1; j++ {
			for k := j + 1; k < n; k++ {
				if e.HasCommonPoint(uniqueLines[i], uniqueLines[j]) &&
					e.HasCommonPoint(uniqueLines[j], uniqueLines[k]) &&
					e.HasCommonPoint(uniqueLines[k], uniqueLines[i]) {

					//triangles = append(triangles, Triangle{
					//	A: uniqueLines[i],
					//	B: uniqueLines[j],
					//	C: uniqueLines[k],
					//})
				}
			}
		}
	}

	return triangles
}

// HasCommonPoint checks if two lines share a common endpoint
func (e *Game) HasCommonPoint(line1, line2 Line) bool {
	return line1.A.Equal(geom.XY, *line2.A) || line1.A.Equal(geom.XY, *line2.B) ||
		line1.B.Equal(geom.XY, *line2.A) || line1.B.Equal(geom.XY, *line2.B)
}
