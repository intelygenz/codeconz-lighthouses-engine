package player

import (
	"context"
	"fmt"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"github.com/spf13/viper"
	"github.com/twpayne/go-geom"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func (p *Player) RequestNewTurn(t Turn) (*Action, error) {
	fmt.Printf("Asking to Player %d to address %v\n", p.ID, p.ServerAddress)

	grpcOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
	grpcClient, err := grpc.NewClient(p.ServerAddress, grpcOpt)
	if err != nil {
		panic(err)
	}

	npjc := coms.NewGameServiceClient(grpcClient)

	ctx, cancel := context.WithTimeout(context.Background(), viper.GetDuration("game.turn_request_timeout"))
	defer cancel()

	playerAction, err := npjc.Turn(ctx, &coms.NewTurn{
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
	for _, lighthouse := range t.Lighthouses {
		haveKey := false
		connections := make([]*coms.Position, 0)
		for _, lKey := range p.LighthouseKeys {
			if lKey.Position.Equal(geom.XY, lighthouse.Position) {
				haveKey = true
			}

			connections = append(connections, &coms.Position{
				X: int32(lKey.Position.X()),
				Y: int32(lKey.Position.Y()),
			})
		}

		lighthouses = append(lighthouses, &coms.Lighthouse{
			Position: &coms.Position{
				X: int32(lighthouse.Position.X()),
				Y: int32(lighthouse.Position.Y()),
			},
			Owner:       int32(lighthouse.Owner),
			Energy:      int32(lighthouse.Energy),
			Connections: connections,
			HaveKey:     haveKey,
		})
	}
	return lighthouses
}
