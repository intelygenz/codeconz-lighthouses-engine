package player

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"
	"github.com/twpayne/go-geom"
)

const (
	ActionPass = iota
	ActionMove
	ActionAttack
	ActionConnect
)

type Action struct {
	Action      int
	Destination geom.Coord
	Energy      int
}

type Turn struct {
	Position    geom.Coord
	Score       int
	Energy      int
	View        [][]int
	Lighthouses []*lighthouse.Lighthouse
}

type Player struct {
	ServerAddress  string
	ID             int
	Name           string
	Score          int
	Energy         int
	Position       geom.Coord
	LighthouseKeys []*lighthouse.Lighthouse
}

func NewPlayer(serverAddress string, name string) *Player {
	return &Player{
		ServerAddress:  serverAddress,
		ID:             -1,
		Name:           name,
		Score:          0,
		Energy:         0,
		Position:       geom.Coord{0, 0},
		LighthouseKeys: make([]*lighthouse.Lighthouse, 0),
	}
}

func (p *Player) SetPosition(x, y int) {
	p.Position = geom.Coord{float64(x), float64(y)}
}

func (p *Player) GetPosition() (x, y int) {
	return int(p.Position.X()), int(p.Position.Y())
}

func (p *Player) AddLighthouseKey(lighthouse lighthouse.Lighthouse) {
	p.LighthouseKeys = append(p.LighthouseKeys, &lighthouse)
}

func (p *Player) RemoveLighthouseKey(lighthouse lighthouse.Lighthouse) {
	for i, l := range p.LighthouseKeys {
		if l.Position.Equal(geom.XY, lighthouse.Position) {
			p.LighthouseKeys = append(p.LighthouseKeys[:i], p.LighthouseKeys[i+1:]...)
			break
		}
	}
}
