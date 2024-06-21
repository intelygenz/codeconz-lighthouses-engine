package handler

import (
	"context"
	"fmt"
	coms2 "github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	timeoutToResponse = 100 * time.Millisecond
)

type GameClient struct {
}

func (gc *GameClient) RequestTurn(p *coms2.NewPlayer, t *coms2.NewTurn) *coms2.NewAction {
	fmt.Printf("Asking to Player %s to address %v\n", p.Name, p.ServerAddress)

	grpcOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
	grpcClient, err := grpc.NewClient(p.ServerAddress, grpcOpt)
	if err != nil {
		panic(err)
	}

	npjc := coms2.NewGameServiceClient(grpcClient)

	ctx, cancel := context.WithTimeout(context.Background(), timeoutToResponse)
	defer cancel()

	playerAction, err := npjc.Turn(ctx, t)
	if err != nil {
		fmt.Println("Error requesting turn", err)
	}
	return playerAction
}
