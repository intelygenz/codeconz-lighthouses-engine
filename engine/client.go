package engine

import (
	"context"
	"fmt"
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/coms"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	timeoutToResponse = 100 * time.Millisecond
)

type GameClient struct {
}

func (gc *GameClient) requestTurn(p *coms.NewPlayer, t *coms.NewTurn) *coms.NewAction {
	fmt.Printf("Asking to Player %s to address %v\n", p.Name, p.ServerAddress)

	grpcOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
	grpcClient, err := grpc.NewClient(p.ServerAddress, grpcOpt)
	if err != nil {
		panic(err)
	}

	npjc := coms.NewGameServiceClient(grpcClient)

	ctx, cancel := context.WithTimeout(context.Background(), timeoutToResponse)
	defer cancel()

	playerAction, err := npjc.Turn(ctx, t)
	if err != nil {
		panic(err)
	}
	return playerAction
}
