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

	lighthouses := make([]*coms.Lighthouse, 0)
	for _, lighthouse := range t.Lighthouses {
		lighthouses = append(lighthouses, &coms.Lighthouse{
			Position: &coms.Position{
				X: int32(lighthouse.Position.X()),
				Y: int32(lighthouse.Position.Y()),
			},
			Owner:  int32(lighthouse.Owner),
			Energy: int32(lighthouse.Energy),
			//Connections: lighthouse.Connections, // TODO map connections
			//HaveKey: lighthouse.HaveKey, // TODO map have key
		})
	}

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

	ctx, cancel := context.WithTimeout(context.Background(), viper.GetDuration("game.turn_request_timeout"))
	defer cancel()

	playerAction, err := npjc.Turn(ctx, &coms.NewTurn{
		Position: &coms.Position{
			X: int32(t.Position.X()),
			Y: int32(t.Position.Y()),
		},
		Score:       int32(t.Score),
		Energy:      int32(t.Energy),
		View:        playerView,
		Lighthouses: lighthouses,
	})
	if err != nil {
		fmt.Println("Error requesting turn", err)
		return nil, err
	}

	return &Action{
		Action:      int(playerAction.Action),
		Destination: geom.Coord{float64(playerAction.Destination.X), float64(playerAction.Destination.Y)},
		Energy:      int(playerAction.Energy),
	}, err
}
