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
	timeoutToResponse = 1 * time.Second
)

type BotGameTurn struct {
	turn   *coms.NewTurn
	action *coms.NewAction
}

type BotGame struct {
	initialState *coms.NewPlayerInitialState
	turnStates   []BotGameTurn
}

func (bg *BotGame) NewTurnAction(turn *coms.NewTurn) coms.NewAction {

	action := coms.NewAction{
		Action: coms.Action_MOVE,
		Destination: &coms.Position{
			X: int32(rand.Intn(10)),
			Y: int32(rand.Intn(10)),
		},
	}

	bgt := BotGameTurn{
		turn:   turn,
		action: &action,
	}
	bg.turnStates = append(bg.turnStates, bgt)

	return action
}

type BotComs struct {
	botName                      string
	myAddress, gameServerAddress string
	initialState                 *coms.NewPlayerInitialState
}

func (ps *BotComs) waitToJoinGame() {
	grpcOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
	grpcClient, err := grpc.NewClient(ps.gameServerAddress, grpcOpt)
	if err != nil {
		fmt.Printf("grpc client ERROR: %v\n", err)
		panic("could not create a grpc client")
	}

	npjc := coms.NewGameServiceClient(grpcClient)

	player := &coms.NewPlayer{
		Name:          ps.botName,
		ServerAddress: ps.myAddress,
	}

	for {
		ctx, cancel := context.WithTimeout(context.Background(), timeoutToResponse)
		initialState, err := npjc.Join(ctx, player)
		// time.Sleep(timeoutToResponse)
		if err != nil {
			fmt.Printf("could not join game ERROR: %v\n", err)
			cancel()
			continue
		} else {
			fmt.Println("Joined game with", initialState)
			ps.initialState = initialState
			break
		}
	}

}

func (ps *BotComs) startListening() {
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

func (gs *ClientServer) Turn(ctx context.Context, turn *coms.NewTurn) (*coms.NewAction, error) {
	bg := &BotGame{}

	action := bg.NewTurnAction(turn)

	return &action, nil
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

	bot := &BotComs{
		botName:           *botName,
		myAddress:         *listenAddress,
		gameServerAddress: *gameServerAddress,
	}

	bot.waitToJoinGame()
	fmt.Println("Received message from server", bot.initialState)
	bot.startListening()

}
