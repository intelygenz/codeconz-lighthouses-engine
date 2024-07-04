package island

import (
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/cell"
	"github.com/twpayne/go-geom"
)

type Island struct {
	cell.Cell
	Energy int
}

func (i Island) GetX() int {
	return int(i.Location.X())
}

func (i Island) GetY() int {
	return int(i.Location.Y())
}

func (i Island) GetType() cell.CellType {
	return i.Type
}

func NewIslandCell(x, y int) *Island {
	return &Island{
		Cell: cell.Cell{
			Location: geom.Coord{float64(x), float64(y)},
			Type:     cell.IslandCell,
		},
		Energy: 0,
	}
}
