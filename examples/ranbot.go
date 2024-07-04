package main

import (
	"flag"
	"fmt"
	coms2 "github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"google.golang.org/grpc/status"
	"net"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	timeoutToResponse = 1 * time.Second
)

type BotGameTurn struct {
	turn   *coms2.NewTurn
	action *coms2.NewAction
}

type BotGame struct {
	initialState *coms2.NewPlayerInitialState
	turnStates   []BotGameTurn
}

func (bg *BotGame) NewTurnAction(turn *coms2.NewTurn) *coms2.NewAction {
	position := &coms2.Position{
		X: turn.Position.X + 1,
		Y: turn.Position.Y + 1,
	}
	action := &coms2.NewAction{
		Action:      coms2.Action_MOVE,
		Destination: position,
	}

	bgt := BotGameTurn{
		turn:   turn,
		action: action,
	}
	bg.turnStates = append(bg.turnStates, bgt)

	return action
}

type BotComs struct {
	botName                      string
	myAddress, gameServerAddress string
	initialState                 *coms2.NewPlayerInitialState
}

func (ps *BotComs) waitToJoinGame() {
	grpcOpt := grpc.WithTransportCredentials(insecure.NewCredentials())
	grpcClient, err := grpc.NewClient(ps.gameServerAddress, grpcOpt)
	if err != nil {
		fmt.Printf("grpc client ERROR: %v\n", err)
		panic("could not create a grpc client")
	}

	npjc := coms2.NewGameServiceClient(grpcClient)

	player := &coms2.NewPlayer{
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

	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(UnaryLoggingInterceptor),
		grpc.StreamInterceptor(StreamLoggingInterceptor),
	)
	cs := &ClientServer{}
	coms2.RegisterGameServiceServer(grpcServer, cs)

	if err := grpcServer.Serve(lis); err != nil {
		panic(err)
	}
}

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

type ClientServer struct{}

func (gs *ClientServer) Join(ctx context.Context, req *coms2.NewPlayer) (*coms2.NewPlayerInitialState, error) {
	return nil, fmt.Errorf("game server does not implement Join sercvice")
}

func (gs *ClientServer) Turn(ctx context.Context, turn *coms2.NewTurn) (*coms2.NewAction, error) {
	bg := &BotGame{}

	action := bg.NewTurnAction(turn)

	return action, nil
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
