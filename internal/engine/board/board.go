package board

import (
	"bufio"
	"fmt"
	"github.com/fatih/color"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/cell"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/island"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/twpayne/go-geom"
	"github.com/twpayne/go-geom/xy"
	"log"
	"math"
	"math/rand"
	"os"
	"os/exec"
)

const (
	PlayerInitialPositionRune = '0'
	WaterCellRune             = 'x'
	IslandCellRune            = ' '
	LighthouseCellRune        = '!'
)

type BoardI interface {
	GetLightHouses() []lighthouse.Lighthouse
	GetPlayableMap() [][]bool
	GetRandomIslandLocation() geom.Coord
	CalcEnergy()
	PrettyPrintBoolMap()
	PrettyPrintMap(players []*player.Player)
}

type CellI interface {
	GetX() int
	GetY() int
	GetType() cell.CellType
}

type Board struct {
	Width  int
	Height int
	Cells  [][]CellI
}

func NewBoard(boardPath string) BoardI {
	board := Board{}
	board.load(boardPath)

	return &board
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
				m.Cells[i][j] = island.NewIslandCell(i, j)
			case WaterCellRune:
				m.Cells[i][j] = cell.NewEmptyCell(i, j)
			case LighthouseCellRune:
				m.Cells[i][j] = lighthouse.NewLightHouseCell(i, j)
			default:
				// These are supposed to be initial player Cells
				m.Cells[i][j] = cell.NewEmptyCell(i, j)
			}
		}
	}
}

func (m *Board) GetLightHouses() []lighthouse.Lighthouse {
	var lighthouses []lighthouse.Lighthouse
	for i := 0; i < m.Height; i++ {
		for j := 0; j < m.Width; j++ {
			if m.Cells[i][j].GetType() == cell.LighthouseCell {
				lighthouses = append(lighthouses, *m.Cells[i][j].(*lighthouse.Lighthouse))
			}
		}
	}
	return lighthouses
}

func (m *Board) GetPlayableMap() [][]bool {
	playableMap := make([][]bool, m.Height)
	for i := 0; i < m.Height; i++ {
		playableMap[i] = make([]bool, m.Width)
		for j := 0; j < m.Width; j++ {
			if m.isPlayableCell(m.Cells[i][j]) {
				playableMap[i][j] = true
			}
		}
	}
	return playableMap
}

func (m *Board) isPlayableCell(c CellI) bool {
	return c.GetType() == cell.IslandCell || c.GetType() == cell.LighthouseCell
}

func (m *Board) GetRandomIslandLocation() geom.Coord {
	for {
		i := rand.Intn(m.Height)
		j := rand.Intn(m.Width)
		if m.Cells[i][j].GetType() == cell.IslandCell {
			return m.Cells[i][j].(*island.Island).Location
		}
	}
}

func (m *Board) CalcEnergy() {

	//Calculate energy for the island and lighthouse cells
	for i := 0; i < m.Height; i++ {
		for j := 0; j < m.Width; j++ {
			if m.Cells[i][j].GetType() == cell.IslandCell {
				islandCell := m.Cells[i][j].(*island.Island)

				// calculate the energy of the island based on the formula: energia += floor(5 - distancia_a_faro)
				for _, lighthouse := range m.GetLightHouses() {
					distance := xy.Distance(islandCell.Location, lighthouse.Location)
					islandCell.Energy += int(math.Max(math.Floor(5-distance), 0))
				}

				// Set max energy to 100
				if islandCell.Energy > 100 {
					islandCell.Energy = 100
				}
			}

			// Remove energy from lighthouses
			if m.Cells[i][j].GetType() == cell.LighthouseCell {
				lighthouseCell := m.Cells[i][j].(*lighthouse.Lighthouse)
				lighthouseCell.Energy -= 10

				// Set min energy to 0
				if lighthouseCell.Energy <= 0 {
					lighthouseCell.Energy = 0

					//TODO: remove connections and the ownership of the lighthouse
				}
			}
		}
	}

	//TODO: Give energy to players, maybe not here
	/*for _, player := range m. {

	}*/
}

func (m *Board) PrettyPrintBoolMap() {
	playableMap := m.GetPlayableMap()
	for i := 0; i < m.Height; i++ {
		for j := 0; j < m.Width; j++ {
			// check if the cell is an island
			switch playableMap[i][j] {
			case true:
				color.New(color.BgGreen).Print(" 1 ")
			case false:
				color.New(color.BgRed).Print(" 0 ")
			}
		}
		fmt.Println()
	}
}

func (m *Board) PrettyPrintMap(players []*player.Player) {
	cmd := exec.Command("clear") //Linux example, its tested
	cmd.Stdout = os.Stdout
	cmd.Run()

	fmt.Println()
	for i := 0; i < m.Height; i++ {
		for j := 0; j < m.Width; j++ {
			// check if the cell is an island
			for _, p := range players {
				if p.Position.Equal(geom.XY, geom.Coord{float64(m.Cells[i][j].GetX()), float64(m.Cells[i][j].GetY())}) {
					color.New(color.BgMagenta).Print(fmt.Sprintf("[%d]", p.ID))
					break
				}
			}

			switch m.Cells[i][j].GetType() {
			case cell.IslandCell:
				color.New(color.BgBlue).Print(fmt.Sprintf("|%d|", m.Cells[i][j].(*island.Island).Energy))
			case cell.LighthouseCell:
				color.New(color.BgYellow).Print(fmt.Sprintf("|%d|", m.Cells[i][j].(*lighthouse.Lighthouse).Energy))
			case cell.WaterCell:
				color.New(color.BgBlack).Print(" x ")
			}
		}
		fmt.Println()
	}
}
