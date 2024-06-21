package board

import "github.com/twpayne/go-geom"

type Lighthouse struct {
	Island
	Owner       int
	Connections []Lighthouse
}

func (l Lighthouse) GetX() int {
	return int(l.Location.X())
}

func (l Lighthouse) GetY() int {
	return int(l.Location.Y())
}

func (l Lighthouse) GetType() CellType {
	return l.Type
}

func NewLightHouseCell(x, y int) Lighthouse {
	return Lighthouse{
		Island: Island{
			Cell: Cell{
				Location: geom.Coord{float64(x), float64(y)},
				Type:     LighthouseCell,
			},
			Energy: 0,
		},
		Owner:       -1,
		Connections: make([]Lighthouse, 0),
	}
}
