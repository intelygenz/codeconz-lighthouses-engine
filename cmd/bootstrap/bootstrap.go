package bootstrap

import (
	"fmt"
	"net"
	"os"
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

	if viper.GetString("game.listen_address") == "" {
		panic("addr is required")
	}

	fmt.Println("Game server starting on", viper.GetString("game.listen_address"))

	lis, err := net.Listen("tcp", viper.GetString("game.listen_address"))
	if err != nil {
		panic(err)
	}

	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(handler.UnaryLoggingInterceptor),
		grpc.StreamInterceptor(handler.StreamLoggingInterceptor),
	)

	ge := game.NewGame(viper.GetString("game.board_path"), viper.GetInt("game.turns"))
	gs := handler.NewGameServer(ge)

	coms.RegisterGameServiceServer(grpcServer, gs)

	go func() {
		<-time.After(viper.GetDuration("game.join_timeout"))
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

	viper.AddConfigPath("./")
	viper.SetConfigName("cfg")
	viper.SetConfigType("yaml")

	err := viper.ReadInConfig()
	if err != nil {
		os.Exit(1)
	}
}
