package cell

import (
	"github.com/twpayne/go-geom"
)

type CellType int

const (
	WaterCell CellType = iota
	IslandCell
)

type Cell struct {
	Position geom.Coord
	Type     CellType
}

func (c Cell) GetX() int {
	return int(c.Position.X())
}

func (c Cell) GetY() int {
	return int(c.Position.Y())
}

func (c Cell) GetPosition() geom.Coord {
	return c.Position
}

func (c Cell) GetType() CellType {
	return c.Type
}

func NewEmptyCell(x, y int) *Cell {
	return &Cell{
		Position: geom.Coord{float64(x), float64(y)},
		Type:     WaterCell,
	}
}
