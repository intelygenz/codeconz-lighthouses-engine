package handler

import (
	"context"
	"fmt"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/game"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"strconv"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

func UnaryLoggingInterceptor(
	ctx context.Context,
	req interface{},
	info *grpc.UnaryServerInfo,
	handler grpc.UnaryHandler,
) (resp interface{}, err error) {
	start := time.Now()
	resp, err = handler(ctx, req)
	duration := time.Since(start)
	st, _ := status.FromError(err)
	fmt.Printf("unary call: %s, Duration: %v, Error: %v\n", info.FullMethod, duration, st.Message())
	return resp, err
}

func StreamLoggingInterceptor(
	srv interface{},
	ss grpc.ServerStream,
	info *grpc.StreamServerInfo,
	handler grpc.StreamHandler,
) error {
	start := time.Now()
	err := handler(srv, ss)
	duration := time.Since(start)
	st, _ := status.FromError(err)
	fmt.Printf("stream call: %s, Duration: %v, Error: %v\n", info.FullMethod, duration, st.Message())
	return err
}

type GameServer struct {
	game game.GameI
}

func NewGameServer(ge game.GameI) *GameServer {
	return &GameServer{
		game: ge,
	}
}

func (gs *GameServer) Join(ctx context.Context, req *coms.NewPlayer) (*coms.NewPlayerInitialState, error) {
	fmt.Printf("New player ask to join %s\n", req.Name)

	playerID, err := strconv.Atoi(req.Name)
	if err != nil {
		fmt.Printf("Error converting player ID %s\n", err)
		return nil, err
	}

	np := player.NewPlayer(req.ServerAddress, playerID)

	err = gs.game.AddNewPlayer(*np)
	if err != nil {
		fmt.Printf("Error adding new player %s\n", err)
		return nil, err
	}

	fmt.Printf("New player joined %d\n", np.ID)

	mapper := Mapper{}

	return mapper.MapPlayerInitialStateToComPlayerInitialState(gs.game.CreateInitialState(*np)), nil //TODO cannot do this as we don't have the players information yet
}

func (gs *GameServer) Turn(ctx context.Context, req *coms.NewTurn) (*coms.NewAction, error) {
	return nil, fmt.Errorf("game server does not implement Turn sercvice")
}
