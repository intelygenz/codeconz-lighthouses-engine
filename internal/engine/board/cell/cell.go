package cell

import (
	"github.com/twpayne/go-geom"
)

type CellType int

const (
	WaterCell CellType = iota
	IslandCell
	LighthouseCell
)

type Cell struct {
	Location geom.Coord
	Type     CellType
}

func (c Cell) GetX() int {
	return int(c.Location.X())
}

func (c Cell) GetY() int {
	return int(c.Location.Y())
}

func (c Cell) GetType() CellType {
	return c.Type
}

func NewEmptyCell(x, y int) *Cell {
	return &Cell{
		Location: geom.Coord{float64(x), float64(y)},
		Type:     WaterCell,
	}
}
