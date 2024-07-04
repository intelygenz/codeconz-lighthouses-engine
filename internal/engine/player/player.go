package player

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/lighthouse"
	"github.com/twpayne/go-geom"
)

const (
	ActionPass = iota
	ActionMove
	ActionAttack
	ActionConnect
)

type Action struct {
	Action      int32
	Destination geom.Coord
	Energy      int32
}

type Turn struct {
	Position    geom.Coord
	Score       int32
	Energy      int32
	View        [][]int32
	Lighthouses []geom.Coord
}

type Player struct {
	ServerAddress  string
	ID             int
	Score          int
	Energy         int
	Position       geom.Coord
	LighthouseKeys []lighthouse.Lighthouse
}

func NewPlayer(serverAddress string, id int) *Player {
	return &Player{
		ServerAddress:  serverAddress,
		ID:             id,
		Score:          0,
		Energy:         0,
		Position:       geom.Coord{0, 0},
		LighthouseKeys: make([]lighthouse.Lighthouse, 0),
	}
}

func (p *Player) SetPosition(x, y int) {
	p.Position = geom.Coord{float64(x), float64(y)}
}

func (p *Player) GetPosition() (x, y int) {
	return int(p.Position.X()), int(p.Position.Y())
}

func (p *Player) AddLighthouseKey(lighthouse lighthouse.Lighthouse) {
	p.LighthouseKeys = append(p.LighthouseKeys, lighthouse)
}
