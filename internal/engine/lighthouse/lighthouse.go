package lighthouse

import (
	"fmt"
	"github.com/twpayne/go-geom"
)

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

func (l *Lighthouse) Connect(lighthouse *Lighthouse) error {
	if l.Owner != lighthouse.Owner {
		return fmt.Errorf("lighthouses must have the same owner")
	}

	// check if the connection already contains the lighthouse
	for _, conn := range l.Connections {
		if conn.Position.Equal(geom.XY, lighthouse.Position) {
			return fmt.Errorf("lighthouse already connected")
		}
	}

	l.Connections = append(l.Connections, *lighthouse)
	lighthouse.Connections = append(lighthouse.Connections, *l)

	return nil
}
