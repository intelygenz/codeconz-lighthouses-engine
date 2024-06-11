package main

import (
	"flag"
	"fmt"
	"math/rand"
	"net"
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/coms"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type GameState struct {
	botName                      string
	myAddress, gameServerAddress string
	initialState                 *coms.NewPlayerInitialState
}

func askToJoinGame(gs *GameState) {
	grpcOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
	grpcClient, err := grpc.NewClient(gs.gameServerAddress, grpcOpt)
	if err != nil {
		panic(err)
	}

	npjc := coms.NewGameServiceClient(grpcClient)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	player := &coms.NewPlayer{
		Name:          gs.botName,
		ServerAddress: gs.myAddress,
	}
	initialState, err := npjc.Join(ctx, player)
	if err != nil {
		panic(err)
	}

	gs.initialState = initialState
}

type ClientServer struct{}

// Join(context.Context, *NewPlayer) (*NewPlayerAccepted, error)
func (gs *ClientServer) Join(ctx context.Context, req *coms.NewPlayer) (*coms.NewPlayerInitialState, error) {
	return nil, fmt.Errorf("game server does not implement Join sercvice")
}

func (gs *ClientServer) Turn(ctx context.Context, req *coms.NewTurn) (*coms.NewAction, error) {
	nt := req
	fmt.Println("Received new turn", nt)

	randomAction := &coms.NewAction{
		Action: coms.Action_MOVE,
		Destination: &coms.Position{
			X: int32(rand.Intn(10)),
			Y: int32(rand.Intn(10)),
		},
	}
	return randomAction, nil
}

func startListening(gs *GameState) {

	lis, err := net.Listen("tcp", gs.myAddress)
	if err != nil {
		panic(err)
	}

	grpcServer := grpc.NewServer()
	cs := &ClientServer{}
	coms.RegisterGameServiceServer(grpcServer, cs)

	if err := grpcServer.Serve(lis); err != nil {
		panic(err)
	}
}

func main() {
	botName := flag.String("bn", "", "bot name")
	listenAddress := flag.String("la", "", "my listen address")
	gameServerAddress := flag.String("gs", "", "game server address")
	flag.Parse()

	if *botName == "" {
		panic("bot name is required")
	}
	if *listenAddress == "" {
		panic("listen address is required")
	}
	if *gameServerAddress == "" {
		panic("game server address is required")
	}

	gs := &GameState{
		botName:           *botName,
		myAddress:         *listenAddress,
		gameServerAddress: *gameServerAddress,
	}

	askToJoinGame(gs)

	fmt.Println("Received message from server", gs.initialState)

	startListening(gs)

}
