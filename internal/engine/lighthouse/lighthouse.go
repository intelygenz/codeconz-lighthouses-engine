package lighthouse

import "github.com/twpayne/go-geom"

type Lighthouse struct {
	Position    geom.Coord
	Energy      int
	Owner       int
	Connections []Lighthouse
}

func NewLightHouse(x, y int) *Lighthouse {
	return &Lighthouse{
		Position:    geom.Coord{float64(x), float64(y)},
		Energy:      0,
		Owner:       -1,
		Connections: make([]Lighthouse, 0),
	}
}
