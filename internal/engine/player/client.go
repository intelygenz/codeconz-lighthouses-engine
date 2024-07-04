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
	for _, lighthouse := range p.LighthouseKeys {
		lighthouses = append(lighthouses, &coms.Lighthouse{
			Position: &coms.Position{
				X: int32(lighthouse.Location.X()),
				Y: int32(lighthouse.Location.Y()),
			},
			// TODO add the rest of the fields
		})
	}

	// TODO reomve this sample
	sampleRows := make([]*coms.MapRow, 0)
	sampleRows = append(sampleRows, &coms.MapRow{Row: []int32{0, 1, 0, 1}})
	sampleRows = append(sampleRows, &coms.MapRow{Row: []int32{0, 1, 1, 1}})
	sampleRows = append(sampleRows, &coms.MapRow{Row: []int32{0, 0, 0, 1}})

	ctx, cancel := context.WithTimeout(context.Background(), viper.GetDuration("game.turn_request_timeout"))
	defer cancel()

	playerAction, err := npjc.Turn(ctx, &coms.NewTurn{
		Position: &coms.Position{
			X: int32(p.Position.X()),
			Y: int32(p.Position.Y()),
		},
		Score:       int32(p.Score),
		Energy:      int32(p.Energy),
		View:        sampleRows, // TODO generate turn view
		Lighthouses: lighthouses,
	})
	if err != nil {
		fmt.Println("Error requesting turn", err)
		return nil, err
	}

	return &Action{
		Action:      int32(playerAction.Action),
		Destination: geom.Coord{float64(playerAction.Destination.X), float64(playerAction.Destination.Y)},
		Energy:      playerAction.Energy,
	}, err
}
