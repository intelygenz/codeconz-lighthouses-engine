package bootstrap

import (
	"fmt"
	"net"
	"time"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/game"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/handler/coms"
	"github.com/spf13/viper"
	"google.golang.org/grpc"
)

type Bootstrap struct{}

func NewBootstrap() *Bootstrap {
	return &Bootstrap{}
}

func (b *Bootstrap) Run() {
	b.initializeConfiguration()

	if viper.GetString("listen_address") == "" {
		panic("addr is required")
	}

	fmt.Println("Game server starting on", viper.GetString("listen_address"))

	lis, err := net.Listen("tcp", viper.GetString("listen_address"))
	if err != nil {
		panic(err)
	}

	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(handler.UnaryLoggingInterceptor),
		grpc.StreamInterceptor(handler.StreamLoggingInterceptor),
	)

	ge := game.NewGame(viper.GetString("board_path"), viper.GetInt("turns"))
	gs := handler.NewGameServer(ge)

	coms.RegisterGameServiceServer(grpcServer, gs)

	go func() {
		<-time.After(viper.GetDuration("join_timeout"))
		grpcServer.Stop()
	}()

	if err := grpcServer.Serve(lis); err != nil {
		panic(err)
	}

	fmt.Println("players joining is closed.")

	if len(ge.GetPlayers()) == 0 {
		panic("No players joined, game did not start.")
	}

	fmt.Println("sending initial state to all players.")
	ge.SendInitialState()

	fmt.Println("Now the game starts!!!")
	ge.StartGame()
	fmt.Println("Game finished!!!")
}

func (b *Bootstrap) initializeConfiguration() {
	viper.AutomaticEnv()

	viper.SetDefault("listen_address", ":50051")
	viper.SetDefault("board_path", "./maps/island.txt")
	viper.SetDefault("turns", 100)
	viper.SetDefault("join_timeout", 5*time.Second)
	viper.SetDefault("turn_request_timeout", 10*time.Millisecond)
	viper.SetDefault("verbosity", true)
	viper.SetDefault("time_between_rounds", 0*time.Second)

	fmt.Println("Loaded configuration:")
	for _, key := range viper.AllKeys() {
		fmt.Println(key + " : " + viper.GetString(key))
	}
}
