package board

import (
	"bufio"
	"fmt"
	"github.com/fatih/color"
	"log"
	"os"
)

type Board struct {
	Width  int
	Height int
	Cells  [][]CellI
}

type CellType int

const (
	WaterCell CellType = iota
	IslandCell
	LighthouseCell
)

const (
	PlayerInitialPositionRune = '0'
	WaterCellRune             = 'x'
	IslandCellRune            = ' '
	LighthouseCellRune        = '!'
)

type CellI interface {
	GetX() int
	GetY() int
	GetType() CellType
}

func NewBoard(islandPath string) Board {
	island := Board{}
	island.load(islandPath)

	return island
}

func (m *Board) load(path string) {
	file, err := os.Open(path)
	if err != nil {
		log.Fatalf("Error al abrir el archivo: %v", err)
	}
	defer file.Close()

	// Leer el archivo línea por línea
	var lines [][]rune
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) > 0 { // Verifica que la línea no esté vacía
			lines = append(lines, []rune(line))
		}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error al leer el archivo: %v", err)
	}

	// Imprimir la matriz de caracteres
	m.Width = len(lines[0])
	m.Height = len(lines)
	m.Cells = make([][]CellI, m.Height)
	for i := range m.Cells {
		m.Cells[i] = make([]CellI, m.Width)
	}

	for i, line := range lines {
		for j, char := range line {
			switch char {
			case IslandCellRune:
				m.Cells[i][j] = NewIslandCell(i, j)
			case WaterCellRune:
				m.Cells[i][j] = NewEmptyCell(i, j)
			case LighthouseCellRune:
				m.Cells[i][j] = NewLightHouseCell(i, j)
			default:
				// These are supposed to be initial player Cells
				m.Cells[i][j] = NewEmptyCell(i, j)
			}
		}
	}
}

func (m *Board) GetLightHouses() []Lighthouse {
	var lighthouses []Lighthouse
	for i := 0; i < m.Height; i++ {
		for j := 0; j < m.Width; j++ {
			if m.Cells[i][j].GetType() == LighthouseCell {
				lighthouses = append(lighthouses, m.Cells[i][j].(Lighthouse))
			}
		}
	}
	return lighthouses
}

func (m *Board) PrettyPrintMap() {
	for i := 0; i < m.Height; i++ {
		for j := 0; j < m.Width; j++ {
			// check if the cell is an island
			switch m.Cells[i][j].GetType() {
			case IslandCell:
				color.New(color.BgBlue).Print("   ")
			case LighthouseCell:
				color.New(color.BgYellow).Print(" ! ")
			case WaterCell:
				color.New(color.BgBlack).Print(" x ")
			}
		}
		fmt.Println()
	}
}
