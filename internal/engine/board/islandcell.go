package board

import "github.com/twpayne/go-geom"

type Island struct {
	Cell
	Energy int
}

func (i Island) GetX() int {
	return int(i.Location.X())
}

func (i Island) GetY() int {
	return int(i.Location.Y())
}

func (i Island) GetType() CellType {
	return i.Type
}

func NewIslandCell(x, y int) *Island {
	return &Island{
		Cell: Cell{
			Location: geom.Coord{float64(x), float64(y)},
			Type:     IslandCell,
		},
		Energy: 0,
	}
}
