package game

import (
	"testing"

	"github.com/stretchr/testify/suite"
	"github.com/twpayne/go-geom"

	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/lighthouse"
	"github.com/jonasdacruz/lighthouses_aicontest/internal/engine/player"
)

// island.txt is the default board_path of the game server
const testBoardPath = "../../../maps/island.txt"

// lighthouses of island.txt used on the tests below
var (
	lighthouse11   = geom.Coord{float64(1), float64(1)}
	lighthouse64   = geom.Coord{float64(6), float64(4)}
	lighthouse92   = geom.Coord{float64(9), float64(2)}
	lighthouse88   = geom.Coord{float64(8), float64(8)}
	lighthouse310  = geom.Coord{float64(3), float64(10)}
	lighthouse814  = geom.Coord{float64(8), float64(14)}
	lighthouse1111 = geom.Coord{float64(11), float64(11)}
)

type ActionsTestSuite struct {
	suite.Suite

	game   *Game
	player *player.Player
}

func (s *ActionsTestSuite) SetupTest() {
	s.game = NewGame(testBoardPath, 1).(*Game)
	s.player = &player.Player{ID: 1, Energy: 100, Position: geom.Coord{float64(0), float64(0)}}
	s.game.players = append(s.game.players, s.player)
}

// ownLighthouse gives the lighthouse placed at position to the test player
func (s *ActionsTestSuite) ownLighthouse(position geom.Coord) *lighthouse.Lighthouse {
	l, err := s.game.gameMap.GetLightHouse(position)
	s.Require().NoError(err)
	l.Owner = s.player.ID
	l.Energy = 100
	return l
}

// connect moves the player to the origin lighthouse, gives it the key of the
// destination one and executes the connect action
func (s *ActionsTestSuite) connect(from, to *lighthouse.Lighthouse) error {
	s.player.Position = from.Position
	s.player.AddLighthouseKey(to)
	return s.game.execPlayerAction(s.player, &player.Action{
		Action:      player.ActionConnect,
		Destination: to.Position,
	})
}

// A player owning three lighthouses must be able to join them in a triangle,
// as the score is calculated from the connections and the triangles it closes
func (s *ActionsTestSuite) TestConnectLighthousesClosingATriangle() {
	first := s.ownLighthouse(lighthouse11)
	second := s.ownLighthouse(lighthouse64)
	third := s.ownLighthouse(lighthouse92)

	s.NoError(s.connect(first, second))
	// the second connection starts on the lighthouse the first one ended at
	s.NoError(s.connect(second, third))
	// the third connection shares a lighthouse with both previous ones
	s.NoError(s.connect(third, first))

	s.Len(first.Connections, 2)
	s.Len(second.Connections, 2)
	s.Len(third.Connections, 2)

	s.game.CalcPlayersScores()

	// 2 points per owned lighthouse plus 2 points per connection, everything
	// above that can only come from the cells inside the closed triangle
	s.Greater(s.player.Score, 3*2+3*2)
}

// Connections that do cross each other are still rejected
func (s *ActionsTestSuite) TestConnectLighthousesCrossingAnotherConnection() {
	origin := s.ownLighthouse(lighthouse88)
	destination := s.ownLighthouse(lighthouse310)

	crossed := s.ownLighthouse(lighthouse64)
	crossing := s.ownLighthouse(lighthouse814)
	s.Require().NoError(crossed.Connect(crossing))

	s.EqualError(s.connect(origin, destination), "connection cannot intersect another connection")
	s.Len(origin.Connections, 0)
}

// A connection running over a third lighthouse is still rejected, even when it
// starts on a lighthouse that already has a connection
func (s *ActionsTestSuite) TestConnectLighthousesOverAnotherLighthouse() {
	origin := s.ownLighthouse(lighthouse88)
	destination := s.ownLighthouse(lighthouse1111)

	// (1,1), (8,8) and (11,11) are aligned, so the connection below would run
	// over the lighthouse at (1,1)
	aligned := s.ownLighthouse(lighthouse11)
	s.Require().NoError(aligned.Connect(origin))

	s.EqualError(s.connect(origin, destination), "connection cannot intersect a lighthouse")
	s.Len(destination.Connections, 0)
}

func TestActionsSuite(t *testing.T) {
	suite.Run(t, new(ActionsTestSuite))
}
