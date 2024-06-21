package main

import _map "github.com/jonasdacruz/lighthouses_aicontest/internal/engine/board"

func main() {
	/*boot := bootstrap.NewBootstrap()

	boot.Run()*/

	island := _map.NewBoard("/Users/dovixman/Workspace/Work/Programs/SecretEvent/lighthouses_aicontest/maps/square_xl.txt")
	island.PrettyPrintMap()
}
