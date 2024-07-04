package lighthouse

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/cell"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/island"
	"github.com/twpayne/go-geom"
)

type Lighthouse struct {
	island.Island
	Owner       int
	Connections []Lighthouse
}

func (l Lighthouse) GetX() int {
	return int(l.Location.X())
}

func (l Lighthouse) GetY() int {
	return int(l.Location.Y())
}

func (l Lighthouse) GetType() cell.CellType {
	return l.Type
}

func NewLightHouseCell(x, y int) *Lighthouse {
	return &Lighthouse{
		Island: island.Island{
			Cell: cell.Cell{
				Location: geom.Coord{float64(x), float64(y)},
				Type:     cell.LighthouseCell,
			},
			Energy: 0,
		},
		Owner:       -1,
		Connections: make([]Lighthouse, 0),
	}
}
