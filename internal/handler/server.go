package handler

import (
	"context"
	"fmt"
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/game"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"

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

func (gs *GameServer) Join(_ context.Context, req *coms.NewPlayer) (*coms.PlayerID, error) {
	fmt.Printf("New player ask to join %s\n", req.Name)

	np := player.NewPlayer(req.ServerAddress, req.GetName())

	err := gs.game.AddNewPlayer(np)
	if err != nil {
		fmt.Printf("Error adding new player %s\n", err)
		return nil, err
	}

	fmt.Printf("New player %s joined with ID %d\n", np.Name, np.ID)

	return &coms.PlayerID{PlayerID: int32(np.ID)}, nil
}

func (gs *GameServer) InitialState(_ context.Context, playerID *coms.PlayerID) (*coms.NewPlayerInitialState, error) {
	fmt.Printf("New player ask for initial state %d\n", playerID.PlayerID)

	p := gs.game.GetPlayerByID(int(playerID.PlayerID))

	if p == nil {
		return nil, fmt.Errorf("player not found")
	}

	mapper := Mapper{}
	playerInitialState := gs.game.CreateInitialState(p)

	return mapper.MapPlayerInitialStateToComPlayerInitialState(playerInitialState), nil
}

func (gs *GameServer) Turn(_ context.Context, _ *coms.NewTurn) (*coms.NewAction, error) {
	return nil, fmt.Errorf("game server does not implement Turn sercvice")
}
