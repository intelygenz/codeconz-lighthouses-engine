package board

import (
	"bufio"
	"fmt"
	"github.com/fatih/color"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/cell"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/island"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/twpayne/go-geom"
	"github.com/twpayne/go-geom/xy"
	"log"
	"math"
	"math/rand"
	"os"
)

const (
	PlayerInitialPositionRune = '0'
	WaterCellRune             = '#'
	IslandCellRune            = ' '
	LighthouseCellRune        = '!'

	maxIslandEnergy = 100
)

type BoardI interface {
	GetLightHouses() []*lighthouse.Lighthouse
	GetLightHouse(position geom.Coord) (*lighthouse.Lighthouse, error)
	GetPlayableMap() [][]bool
	GetPlayerView(player *player.Player) [][]int
	GetRandomPlayerInitialPosition() geom.Coord

	CalcIslandEnergy()
	CalcLighthouseEnergy()
	CalcPlayerEnergy(players []*player.Player, currentPlayer *player.Player)
	GetIslandEnergy() [][]int

	CanMoveTo(coord geom.Coord) bool
	IsLighthouse(position geom.Coord) bool
	IsIsland(position geom.Coord) bool
	IsWater(position geom.Coord) bool

	PrettyPrintBoolMap()
	PrettyPrintMap(players []*player.Player)
}

type CellI interface {
	GetX() int
	GetY() int
	GetPosition() geom.Coord
	GetType() cell.CellType
}

type Board struct {
	Width                  int
	Height                 int
	cells                  [][]CellI
	lighthouses            []*lighthouse.Lighthouse
	playerInitialPositions []*island.Island
}

func NewBoard(boardPath string) BoardI {
	board := Board{
		Width:                  0,
		Height:                 0,
		cells:                  nil,
		playerInitialPositions: make([]*island.Island, 0),
		lighthouses:            make([]*lighthouse.Lighthouse, 0),
	}
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
	m.cells = make([][]CellI, m.Height)
	for i := range m.cells {
		m.cells[i] = make([]CellI, m.Width)
	}

	lId := 1
	for i, line := range lines {
		for j, char := range line {
			switch char {
			case IslandCellRune:
				m.cells[i][j] = island.NewIslandCell(i, j)
			case WaterCellRune:
				m.cells[i][j] = cell.NewEmptyCell(i, j)
			case LighthouseCellRune:
				m.cells[i][j] = island.NewIslandCell(i, j)
				m.lighthouses = append(m.lighthouses, lighthouse.NewLightHouse(lId, i, j))
				lId++
			default:
				m.cells[i][j] = island.NewIslandCell(i, j)
				m.playerInitialPositions = append(m.playerInitialPositions, m.cells[i][j].(*island.Island))
			}
		}
	}
}

func (m *Board) GetLightHouses() []*lighthouse.Lighthouse {
	return m.lighthouses
}

func (m *Board) GetLightHouse(position geom.Coord) (*lighthouse.Lighthouse, error) {
	for _, l := range m.lighthouses {
		if l.Position.Equal(geom.XY, position) {
			return l, nil
		}
	}
	return nil, fmt.Errorf("lighthouse not found")
}

func (m *Board) GetPlayableMap() [][]bool {
	playableMap := make([][]bool, m.Height)
	for i := 0; i < m.Height; i++ {
		playableMap[i] = make([]bool, m.Width)
		for j := 0; j < m.Width; j++ {
			if m.cells[i][j].GetType() == cell.IslandCell {
				playableMap[i][j] = true
			}
		}
	}
	return playableMap
}

func (m *Board) GetRandomPlayerInitialPosition() geom.Coord {
	return m.playerInitialPositions[rand.Intn(len(m.playerInitialPositions))].Position
}

func (m *Board) GetPlayerView(player *player.Player) [][]int {
	view := make([][]int, 7)
	maxX := 6
	maxY := 6

	// Loop from (0, 0) to (maxX, maxY)
	for x := 0; x <= maxX; x++ {
		view[x] = make([]int, 7)
		for y := 0; y <= maxY; y++ {
			// Map the new loop indices back to the original range
			i := x + int(player.Position.X()-3)
			j := y + int(player.Position.Y()-3)

			if i < 0 || i >= m.Height || j < 0 || j >= m.Width {
				view[x][y] = -1
			} else {
				if m.cells[i][j].GetType() == cell.IslandCell &&
					xy.Distance(player.Position, m.cells[i][j].GetPosition()) <= 3 {
					view[x][y] = m.cells[i][j].(*island.Island).Energy
				} else {
					view[x][y] = -1
				}
			}
		}
	}

	return view
}

func (m *Board) CalcIslandEnergy() {
	//Calculate energy for the island and lighthouse cells
	for i := 0; i < m.Height; i++ {
		for j := 0; j < m.Width; j++ {
			if m.cells[i][j].GetType() == cell.IslandCell {
				islandCell := m.cells[i][j].(*island.Island)

				// calculate the energy of the island based on the formula: energia += floor(5 - distancia_a_faro)
				for _, lighthouseCell := range m.GetLightHouses() {
					distance := xy.Distance(islandCell.Position, lighthouseCell.Position)
					islandCell.Energy += int(math.Max(math.Floor(5-distance), 0))
				}

				// Set max energy to 100
				if islandCell.Energy > maxIslandEnergy {
					islandCell.Energy = maxIslandEnergy
				}
			}
		}
	}
}

// get energy for the each cell in the board
func (m *Board) GetIslandEnergy() [][]int {
	islandEnergy := make([][]int, m.Height)
	for i := 0; i < m.Height; i++ {
		islandEnergy[i] = make([]int, m.Width)
		for j := 0; j < m.Width; j++ {
			if m.cells[i][j].GetType() == cell.IslandCell {
				islandCell := m.cells[i][j].(*island.Island)
				islandEnergy[i][j] = islandCell.Energy
			} else {
				islandEnergy[i][j] = 0
			}
		}
	}
	return islandEnergy
}

func (m *Board) CalcLighthouseEnergy() {
	// Remove energy from lighthouses
	for _, lg := range m.lighthouses {
		lg.Energy -= 10

		if lg.Energy <= 0 {
			lg.Energy = 0
			lg.Owner = -1

			if len(lg.Connections) > 0 {
				for _, conn := range lg.Connections {
					if len(conn.Connections) == 1 {
						conn.Connections = make([]*lighthouse.Lighthouse, 0)
						conn.ConnectionsId = make([]int, 0)
					}
					for i, l := range conn.Connections {
						if l.Position.Equal(geom.XY, lg.Position) {
							conn.Connections = append(conn.Connections[:i], conn.Connections[i+1:]...)
							conn.ConnectionsId = append(conn.ConnectionsId[:i], conn.ConnectionsId[i+1:]...)
						}
					}
				}
				// current lighthouse loses all connections
				lg.Connections = make([]*lighthouse.Lighthouse, 0)
				lg.ConnectionsId = make([]int, 0)
			}
		}
	}
}

func (m *Board) CalcPlayerEnergy(players []*player.Player, currentPlayer *player.Player) {
	islandCell := m.cells[int(currentPlayer.Position.X())][int(currentPlayer.Position.Y())].(*island.Island)
	if islandCell.Energy == 0 {
		// energy is already given, do nothing
		return
	}

	playersOnPosition := make([]*player.Player, 0)
	for _, p := range players {
		if currentPlayer.Position.Equal(geom.XY, p.Position) {
			playersOnPosition = append(playersOnPosition, p)
		}
	}
	// calculate how much energy is on the island, divide between all players in the position and give corresponding energy
	energy := int(float64(islandCell.Energy / len(playersOnPosition)))
	for _, p := range playersOnPosition {
		p.Energy += energy
	}
	// energy is 0
	islandCell.Energy = 0
}

func (m *Board) CanMoveTo(coord geom.Coord) bool {

	fmt.Printf("CanMoveTo coord: %v\n", coord)
	fmt.Printf("Height: %d, Width: %d\n", m.Height, m.Width)

	if coord.X() < 0 || coord.X() >= float64(m.Height) || coord.Y() < 0 || coord.Y() >= float64(m.Width) {
		return false
	}

	fmt.Printf("Cell type: %v\n", m.cells[int(coord.X())][int(coord.Y())].GetType())

	return m.cells[int(coord.X())][int(coord.Y())].GetType() == cell.IslandCell
}

func (m *Board) IsLighthouse(position geom.Coord) bool {
	for _, lighthouseCell := range m.lighthouses {
		if lighthouseCell.Position.Equal(geom.XY, position) {
			return true
		}
	}

	return false
}

func (m *Board) IsIsland(position geom.Coord) bool {
	return m.isCellOfType(position, cell.IslandCell)
}

func (m *Board) IsWater(position geom.Coord) bool {
	return m.isCellOfType(position, cell.WaterCell)
}

func (m *Board) isCellOfType(position geom.Coord, cellType cell.CellType) bool {
	if position.X() < 0 || position.X() >= float64(m.Height) || position.Y() < 0 || position.Y() >= float64(m.Width) {
		return false
	}

	return m.cells[int(position.X())][int(position.Y())].GetType() == cellType
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
	fmt.Println()
	for i := 0; i < m.Height; i++ {
		for j := 0; j < m.Width; j++ {
			playerCell := (*player.Player)(nil)

			// check if the cell is an island
			for _, p := range players {
				if p.Position.Equal(geom.XY, geom.Coord{float64(m.cells[i][j].GetX()), float64(m.cells[i][j].GetY())}) {
					playerCell = p
					break
				}
			}

			if playerCell != nil {
				color.New(color.BgMagenta).Print(fmt.Sprintf("[%d]", playerCell.ID))
			} else {
				switch m.cells[i][j].GetType() {
				case cell.IslandCell:
					isLighthouse := false
					isInitialPlayerPosition := false
					for _, lighthouseCell := range m.lighthouses {
						if lighthouseCell.Position.Equal(geom.XY, geom.Coord{float64(m.cells[i][j].GetX()), float64(m.cells[i][j].GetY())}) {
							isLighthouse = true
						}
					}

					if isLighthouse {
						color.New(color.BgYellow).Print(fmt.Sprintf("|%d|", m.cells[i][j].(*island.Island).Energy))
					} else {
						for _, initialPlayerPosition := range m.playerInitialPositions {
							if initialPlayerPosition.Position.Equal(geom.XY, geom.Coord{float64(m.cells[i][j].GetX()), float64(m.cells[i][j].GetY())}) {
								isInitialPlayerPosition = true
							}
						}

						if isInitialPlayerPosition {
							color.New(color.BgGreen).Print(fmt.Sprintf("|%d|", m.cells[i][j].(*island.Island).Energy))
						} else {
							color.New(color.BgBlue).Print(fmt.Sprintf("|%d|", m.cells[i][j].(*island.Island).Energy))
						}
					}
				case cell.WaterCell:
					color.New(color.BgBlack).Print(" x ")
				}
			}
		}
		fmt.Println()
	}
}
