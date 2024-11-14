package lighthouse

import (
	"fmt"

	"github.com/twpayne/go-geom"
)

type Lighthouse struct {
	ID            int           `json:"id"`
	Position      geom.Coord    `json:"position"`
	Energy        int           `json:"energy"`
	Owner         int           `json:"ownerId"`
	Connections   []*Lighthouse `json:"-"`
	ConnectionsId []int         `json:"connections"`
}

func NewLightHouse(lId, x, y int) *Lighthouse {
	return &Lighthouse{
		ID:          lId,
		Position:    geom.Coord{float64(x), float64(y)},
		Energy:      0,
		Owner:       -1,
		Connections: make([]*Lighthouse, 0),
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

	l.Connections = append(l.Connections, lighthouse)
	lighthouse.Connections = append(lighthouse.Connections, l)

	return nil
}

func (l *Lighthouse) Disconnect() error {
	for _, conn := range l.Connections {
		for i, c := range conn.Connections {
			if c.Position.Equal(geom.XY, l.Position) {
				conn.Connections = append(conn.Connections[:i], conn.Connections[i+1:]...)
				break
			}
		}
	}

	l.Connections = make([]*Lighthouse, 0)

	return nil

}

func (l *Lighthouse) GenerateConnectionsId() {
	l.ConnectionsId = make([]int, 0)
	for _, c := range l.Connections {
		l.ConnectionsId = append(l.ConnectionsId, c.ID)
	}
}
