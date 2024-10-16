package game

import (
	boardCell "github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board/cell"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
)

type Mapper struct {
}

func (m *Mapper) MapPlayerInitialStateToComPlayerInitialState(playerInitialState *PlayerInitialState) *coms.NewPlayerInitialState {
	return &coms.NewPlayerInitialState{
		PlayerID:    int32(playerInitialState.PlayerID), // ISSUE: 0 is not send
		PlayerCount: int32(playerInitialState.PlayerCount),
		Position: &coms.Position{
			X: int32(playerInitialState.Position.X()),
			Y: int32(playerInitialState.Position.Y()),
		},
		Map:         m.MapPlayableBoolMapToMapRowList(playerInitialState.Map),
		Lighthouses: m.MapLighthousesToComLighthouses(playerInitialState.Lighthouses),
	}
}

func (m *Mapper) MapPlayableBoolMapToMapRowList(island [][]bool) []*coms.MapRow {
	rows := make([]*coms.MapRow, 0)
	for _, row := range island {
		cells := make([]int32, 0)
		for _, cell := range row {
			if cell {
				cells = append(cells, int32(boardCell.WaterCell))
			} else {
				cells = append(cells, int32(boardCell.IslandCell))
			}
		}
		rows = append(rows, &coms.MapRow{Row: cells})
	}

	return rows
}

func (m *Mapper) MapLighthousesToComLighthouses(lighthouseList []*lighthouse.Lighthouse) []*coms.Lighthouse {
	lighthouses := make([]*coms.Lighthouse, 0)
	for _, l := range lighthouseList {
		lighthouses = append(
			lighthouses,
			&coms.Lighthouse{
				Position: &coms.Position{
					X: int32(l.Position.X()),
					Y: int32(l.Position.Y()),
				},
				Owner: int32(l.Owner),
			},
		)
	}
	return lighthouses
}
