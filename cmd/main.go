package main

import "github.com/jonasdacruz/lighthouses_aicontest/cmd/bootstrap"

func main() {
	boot := bootstrap.NewBootstrap()
	boot.Run()
}
