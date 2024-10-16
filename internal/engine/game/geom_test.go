package game

import (
	"testing"

	"github.com/go-playground/assert/v2"
	"github.com/stretchr/testify/suite"
	"github.com/twpayne/go-geom"
)

type GameTestSuite struct {
	suite.Suite
}

var position00 = geom.Coord{float64(0), float64(0)}
var position03 = geom.Coord{float64(0), float64(3)}
var position30 = geom.Coord{float64(3), float64(0)}
var position33 = geom.Coord{float64(3), float64(3)}

var line00_03 = Line{A: &position00, B: &position03}
var line00_30 = Line{A: &position00, B: &position30}
var line03_30 = Line{A: &position03, B: &position30}
var line03_33 = Line{A: &position03, B: &position33}
var line30_33 = Line{A: &position30, B: &position33}
var line03_00 = Line{A: &position03, B: &position00}
var line30_00 = Line{A: &position30, B: &position00}
var line30_03 = Line{A: &position30, B: &position03}

var testCasesGenerateTrianglesFromLines = []struct {
	name      string
	lines     []Line
	triangles []Triangle
}{
	{
		name:      "empty lines and no triangles",
		lines:     make([]Line, 0),
		triangles: []Triangle{},
	},
	{
		name:      "one line and no triangles",
		lines:     []Line{line00_03},
		triangles: []Triangle{},
	},
	{
		name:      "two lines and no triangles",
		lines:     []Line{line00_03, line00_30},
		triangles: []Triangle{},
	},
	{
		name:      "three lines and one triangle",
		lines:     []Line{line00_03, line00_30, line03_30},
		triangles: []Triangle{{A: position00, B: position03, C: position30}},
	},
	{
		name:      "four lines and one triangle",
		lines:     []Line{line00_03, line00_30, line03_30, line03_33},
		triangles: []Triangle{{A: position00, B: position03, C: position30}},
	},
	{
		name:      "five lines and two triangles",
		lines:     []Line{line00_03, line00_30, line03_30, line03_33, line30_33},
		triangles: []Triangle{{A: position00, B: position03, C: position30}, {A: position03, B: position30, C: position33}},
	},
	{
		name:      "six duplicated lines are only one triangle",
		lines:     []Line{line00_03, line00_30, line03_30, line03_00, line30_00, line30_03},
		triangles: []Triangle{{A: position00, B: position03, C: position30}},
	},
}

func (s *GameTestSuite) TestGenerateTrianglesFromLines() {
	for _, tc := range testCasesGenerateTrianglesFromLines {
		s.Suite.Run(tc.name, func() {
			triangles := GenerateTrianglesFromLines(tc.lines)
			// triangles may be not sorted, cannot assert
			for _, tri := range triangles {
				assert.Equal(s.T(), Contains(tc.triangles, tri), true)
			}
			assert.Equal(s.T(), len(tc.triangles), len(triangles))
		})
	}
}

// Contains
var testCasesContains = []struct {
	name      string
	triangles []Triangle
	triangle  Triangle
	expected  bool
}{
	{
		name:      "empty triangles list",
		triangles: []Triangle{},
		triangle:  Triangle{},
		expected:  false,
	},
	{
		name:      "triangle not in triangles",
		triangles: []Triangle{Triangle{A: position00, B: position03, C: position30}},
		triangle:  Triangle{A: position00, B: position03, C: position33},
		expected:  false,
	},
	{
		name:      "triangle in list of 1 triangle",
		triangles: []Triangle{Triangle{A: position00, B: position03, C: position30}},
		triangle:  Triangle{A: position00, B: position03, C: position30},
		expected:  true,
	},
	{
		name:      "triangle in list of 2 triangles",
		triangles: []Triangle{Triangle{A: position00, B: position03, C: position30}, Triangle{A: position03, B: position30, C: position30}},
		triangle:  Triangle{A: position00, B: position03, C: position30},
		expected:  true,
	},
}

func (s *GameTestSuite) TestContains() {
	for _, tc := range testCasesContains {
		s.Suite.Run(tc.name, func() {
			assert.Equal(s.T(), Contains(tc.triangles, tc.triangle), tc.expected)
		})
	}
}

// HasCommonPoint
var testCasesHasCommonPoint = []struct {
	name     string
	line1    Line
	line2    Line
	expected bool
}{
	{
		name:     "has common point 00_03 and 03_33",
		line1:    line00_03,
		line2:    line03_33,
		expected: true,
	},
	{
		name:     "has common point 00_03 and 00_30",
		line1:    line00_03,
		line2:    line00_30,
		expected: true,
	},
	{
		name:     "not has common point 00_03 and 30_33",
		line1:    line00_03,
		line2:    line30_33,
		expected: false,
	},
}

func (s *GameTestSuite) TestHasCommonPoint() {
	for _, tc := range testCasesHasCommonPoint {
		s.Suite.Run(tc.name, func() {
			assert.Equal(s.T(), HasCommonPoint(tc.line1, tc.line2), tc.expected)
		})
	}
}

// GetCommonPoint
var testCasesGetCommonPoint = []struct {
	name          string
	line1         Line
	line2         Line
	expectedPoint geom.Coord
}{
	{
		name:          "common point from 00_03 and 03_33 is is 03",
		line1:         line00_03,
		line2:         line03_33,
		expectedPoint: position03,
	},
	{
		name:          "common point from 03_33 and 00_03 is is 03",
		line1:         line03_33,
		line2:         line00_03,
		expectedPoint: position03,
	},
	{
		name:          "common point from 00_03 and 00_30 is is 00",
		line1:         line00_03,
		line2:         line00_30,
		expectedPoint: position00,
	},
	{
		name:          "common point from 00_30 and 00_03 is is 00",
		line1:         line00_30,
		line2:         line00_03,
		expectedPoint: position00,
	},
}

func (s *GameTestSuite) TestGetCommonPoint() {
	for _, tc := range testCasesGetCommonPoint {
		s.Suite.Run(tc.name, func() {
			assert.Equal(s.T(), GetCommonPoint(tc.line1, tc.line2), tc.expectedPoint)
		})
	}
}

// NormalizeLine
var testCasesNormalizeLine = []struct {
	name         string
	line         Line
	expectedLine Line
}{
	{
		name:         "line 00_03 is ok",
		line:         line00_03,
		expectedLine: line00_03,
	},
	{
		name:         "line 03_00 is not ok and fixed to 00_03",
		line:         Line{A: &position03, B: &position00},
		expectedLine: line00_03,
	},
	{
		name:         "line 30_00 is not ok and fixed to 00_30",
		line:         Line{A: &position30, B: &position00},
		expectedLine: line00_30,
	},
	{
		name:         "line 03_30 is ok",
		line:         line03_30,
		expectedLine: line03_30,
	},
	{
		name:         "line 30_03 is not ok and fixed to 03_30",
		line:         Line{A: &position30, B: &position03},
		expectedLine: Line{A: &position03, B: &position30},
	},
}

func (s *GameTestSuite) TestNormalizeLine() {
	for _, tc := range testCasesNormalizeLine {
		s.Suite.Run(tc.name, func() {
			assert.Equal(s.T(), NormalizeLine(tc.line), tc.expectedLine)
		})
	}
}

// NormalizeTriangle
var testCasesNormalizeTriangle = []struct {
	name             string
	triangle         Triangle
	expectedTriangle Triangle
}{
	{
		name:             "triangle 00,03,30 is ok",
		triangle:         Triangle{A: position00, B: position03, C: position30},
		expectedTriangle: Triangle{A: position00, B: position03, C: position30},
	},
	{
		name:             "triangle 00,30,03 is not ok and fixed to 00,03,30",
		triangle:         Triangle{A: position00, B: position30, C: position03},
		expectedTriangle: Triangle{A: position00, B: position03, C: position30},
	},
	{
		name:             "triangle 03,00,30, is not ok and fixed to 00,03,30",
		triangle:         Triangle{A: position03, B: position00, C: position30},
		expectedTriangle: Triangle{A: position00, B: position03, C: position30},
	},
	{
		name:             "triangle 03,30,00 is not ok and fixed to 00,03,30",
		triangle:         Triangle{A: position03, B: position30, C: position00},
		expectedTriangle: Triangle{A: position00, B: position03, C: position30},
	},
	{
		name:             "triangle 30,00,03 is not ok and fixed to 00,03,30",
		triangle:         Triangle{A: position30, B: position00, C: position03},
		expectedTriangle: Triangle{A: position00, B: position03, C: position30},
	},
	{
		name:             "triangle 30,03,00 is not ok and fixed to 00,03,30",
		triangle:         Triangle{A: position30, B: position03, C: position00},
		expectedTriangle: Triangle{A: position00, B: position03, C: position30},
	},
}

func (s *GameTestSuite) TestNormalizeTriangle() {
	for _, tc := range testCasesNormalizeTriangle {
		s.Suite.Run(tc.name, func() {
			assert.Equal(s.T(), NormalizeTriangle(tc.triangle.A, tc.triangle.B, tc.triangle.C), tc.expectedTriangle)
		})
	}
}

func TestGameSuite(t *testing.T) {
	suite.Run(t, new(GameTestSuite))
}
