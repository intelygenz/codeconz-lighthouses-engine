package player

import (
	"context"
	"fmt"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"github.com/spf13/viper"
	"github.com/twpayne/go-geom"
)

func (p *Player) RequestNewTurn(t Turn) (*Action, error) {
	fmt.Printf("Asking to Player %d to address %v\n", p.ID, p.ServerAddress)

	ctx, cancel := context.WithTimeout(context.Background(), viper.GetDuration("turn_request_timeout"))
	defer cancel()

	playerAction, err := p.gameServiceClient.Turn(ctx, &coms.NewTurn{
		Position: &coms.Position{
			X: int32(t.Position.X()),
			Y: int32(t.Position.Y()),
		},
		Score:       int32(t.Score),
		Energy:      int32(t.Energy),
		View:        p.getPlayerView(t),
		Lighthouses: p.getLighthouses(t),
	})
	if err != nil {
		fmt.Println("Error requesting turn", err)
		return nil, err
	}

	return p.mapPlayerAction(playerAction), err
}

func (p *Player) mapPlayerAction(playerAction *coms.NewAction) *Action {
	var destination geom.Coord
	if playerAction.GetDestination() != nil {
		destination = geom.Coord{
			float64(playerAction.GetDestination().GetX()),
			float64(playerAction.GetDestination().GetY()),
		}
	}

	return &Action{
		Action:      int(playerAction.GetAction()),
		Destination: destination,
		Energy:      int(playerAction.GetEnergy()),
	}
}

func (p *Player) getPlayerView(t Turn) []*coms.MapRow {
	playerView := make([]*coms.MapRow, 0)
	for _, row := range t.View {
		cells := make([]int32, 0)
		for _, cell := range row {
			cells = append(cells, int32(cell))
		}
		playerView = append(playerView, &coms.MapRow{
			Row: cells,
		})
	}
	return playerView
}

func (p *Player) getLighthouses(t Turn) []*coms.Lighthouse {
	lighthouses := make([]*coms.Lighthouse, 0)
	for _, lg := range t.Lighthouses {
		lighthouses = append(lighthouses, &coms.Lighthouse{
			Position: &coms.Position{
				X: int32(lg.Position.X()),
				Y: int32(lg.Position.Y()),
			},
			Owner:       int32(lg.Owner),
			Energy:      int32(lg.Energy),
			Connections: p.mapConnections(lg.Connections),
			HaveKey:     p.mapHaveKeys(lg),
		})
	}

	return lighthouses
}

func (p *Player) mapConnections(connections []*lighthouse.Lighthouse) []*coms.Position {
	conns := make([]*coms.Position, 0)
	for _, conn := range connections {
		conns = append(conns, &coms.Position{
			X: int32(conn.Position.X()),
			Y: int32(conn.Position.Y()),
		})
	}

	return conns
}

func (p *Player) mapHaveKeys(l *lighthouse.Lighthouse) bool {
	for _, lKey := range p.LighthouseKeys {
		if lKey.Position.Equal(geom.XY, l.Position) {
			return true
		}
	}

	return false
}
