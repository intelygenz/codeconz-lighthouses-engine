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

const (
	timeoutToResponse = 100 * time.Millisecond
)

type BotState struct {
	botName                      string
	myAddress, gameServerAddress string
	initialState                 *coms.NewPlayerInitialState
}

func (ps *BotState) askToJoinGame() {
	grpcOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
	grpcClient, err := grpc.NewClient(ps.gameServerAddress, grpcOpt)
	if err != nil {
		panic(err)
	}

	npjc := coms.NewGameServiceClient(grpcClient)

	ctx, cancel := context.WithTimeout(context.Background(), timeoutToResponse)
	defer cancel()

	player := &coms.NewPlayer{
		Name:          ps.botName,
		ServerAddress: ps.myAddress,
	}
	initialState, err := npjc.Join(ctx, player)
	if err != nil {
		panic(err)
	}

	ps.initialState = initialState
}

func (ps *BotState) startListening() {
	fmt.Println("Starting to listen on", ps.myAddress)

	lis, err := net.Listen("tcp", ps.myAddress)
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

type ClientServer struct{}

func (gs *ClientServer) Join(ctx context.Context, req *coms.NewPlayer) (*coms.NewPlayerInitialState, error) {
	return nil, fmt.Errorf("game server does not implement Join sercvice")
}

func (gs *ClientServer) Turn(ctx context.Context, req *coms.NewTurn) (*coms.NewAction, error) {
	nt := req
	fmt.Printf("Received turn request %s\n", nt)

	randomAction := &coms.NewAction{
		Action: coms.Action_MOVE,
		Destination: &coms.Position{
			X: int32(rand.Intn(10)),
			Y: int32(rand.Intn(10)),
		},
	}
	return randomAction, nil
}

func ensureParams() (botName *string, listenAddress *string, gameServerAddress *string) {
	botName = flag.String("bn", "", "bot name")
	listenAddress = flag.String("la", "", "my listen address")
	gameServerAddress = flag.String("gs", "", "game server address")
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
	return botName, listenAddress, gameServerAddress
}

func main() {
	botName, listenAddress, gameServerAddress := ensureParams()

	bot := &BotState{
		botName:           *botName,
		myAddress:         *listenAddress,
		gameServerAddress: *gameServerAddress,
	}

	bot.askToJoinGame()
	fmt.Println("Received message from server", bot.initialState)
	bot.startListening()

}
