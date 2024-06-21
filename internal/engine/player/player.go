package player

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board"
	"github.com/twpayne/go-geom"
)

type Player struct {
	ServerAddress  string
	ID             int
	Score          int
	Energy         int
	Position       geom.Coord
	LighthouseKeys []board.Lighthouse
}

func NewPlayer(serverAddress string, id int) *Player {
	return &Player{
		ServerAddress:  serverAddress,
		ID:             id,
		Score:          0,
		Energy:         0,
		Position:       geom.Coord{0, 0},
		LighthouseKeys: make([]board.Lighthouse, 0),
	}
}

func (p *Player) SetPosition(x, y int) {
	p.Position = geom.Coord{float64(x), float64(y)}
}

func (p *Player) GetPosition() (x, y int) {
	return int(p.Position.X()), int(p.Position.Y())
}

func (p *Player) AddLighthouseKey(lighthouse board.Lighthouse) {
	p.LighthouseKeys = append(p.LighthouseKeys, lighthouse)
}
