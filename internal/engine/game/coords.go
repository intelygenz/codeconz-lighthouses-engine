package game

import (
	"github.com/twpayne/go-geom"
	"github.com/twpayne/go-geom/sorting"
)

type Line struct {
	A, B *geom.Coord
}

type Triangle struct {
	A, B, C geom.Coord
}

// NormalizeLine ensures that a line is always stored with the smaller point first
func NormalizeLine(line Line) Line {
	if (line.A.X() < line.B.X()) || (line.A.X() == line.B.X() && line.A.Y() < line.B.Y()) {
		return line
	}

	return Line{A: line.B, B: line.A}
}

// NormalizeTriangle ensures that a line is always stored with the smaller point first
func NormalizeTriangle(vA, vB, vC geom.Coord) Triangle {
	if sorting.IsLess2D(vA, vB) {
		if sorting.IsLess2D(vB, vC) {
			return Triangle{
				A: vA,
				B: vB,
				C: vC,
			}
		} else {
			if sorting.IsLess2D(vA, vC) {
				return Triangle{
					A: vA,
					B: vC,
					C: vB,
				}
			} else {
				return Triangle{
					A: vC,
					B: vA,
					C: vB,
				}
			}
		}
	} else {
		if sorting.IsLess2D(vA, vC) {
			if sorting.IsLess2D(vB, vC) {
				return Triangle{
					A: vB,
					B: vA,
					C: vC,
				}
			}
		} else {
			if sorting.IsLess2D(vB, vC) {
				return Triangle{
					A: vB,
					B: vC,
					C: vA,
				}

			} else {
				return Triangle{
					A: vC,
					B: vB,
					C: vA,
				}
			}
		}
	}

	return Triangle{}
}

// GenerateTrianglesFromLines generates all possible triangles from a given array of lines
func GenerateTrianglesFromLines(lines []Line) []Triangle {
	triangles := []Triangle{}

	// Normalize lines to handle duplicates with reversed points
	linesMap := make(map[Line]bool)
	for _, line := range lines {
		normalizedLine := NormalizeLine(line)
		if _, ok := linesMap[normalizedLine]; !ok {
			linesMap[normalizedLine] = true
		}
	}

	// Create a slice of unique lines, commented as it alters order
	var uniqueLines []Line
	// var uniqueLines []Line
	for line := range linesMap {
		uniqueLines = append(uniqueLines, line)
	}
	n := len(uniqueLines)

	// Generate all possible triangles from unique lines
	for i := 0; i < n-2; i++ {
		for j := i + 1; j < n-1; j++ {
			for k := j + 1; k < n; k++ {
				if HasCommonPoint(uniqueLines[i], uniqueLines[j]) &&
					HasCommonPoint(uniqueLines[j], uniqueLines[k]) &&
					HasCommonPoint(uniqueLines[k], uniqueLines[i]) {

					verticeA := GetCommonPoint(uniqueLines[i], uniqueLines[j])
					verticeB := GetCommonPoint(uniqueLines[j], uniqueLines[k])
					verticeC := GetCommonPoint(uniqueLines[k], uniqueLines[i])

					if !verticeA.Equal(geom.XY, verticeB) && !verticeB.Equal(geom.XY, verticeC) && !verticeA.Equal(geom.XY, verticeC) {
						triangle := NormalizeTriangle(verticeA, verticeB, verticeC)
						if !Contains(triangles, triangle) {
							triangles = append(triangles, triangle)
						}
					}
				}
			}
		}
	}
	return triangles
}

// HasCommonPoint checks if two lines share a common endpoint
func HasCommonPoint(line1, line2 Line) bool {
	return line1.A.Equal(geom.XY, *line2.A) || line1.A.Equal(geom.XY, *line2.B) ||
		line1.B.Equal(geom.XY, *line2.A) || line1.B.Equal(geom.XY, *line2.B)
}

func GetCommonPoint(line1, line2 Line) geom.Coord {
	if line1.A.Equal(geom.XY, *line2.A) || line1.B.Equal(geom.XY, *line2.A) {
		return *line2.A
	}
	return *line2.B
}

func Contains(tris []Triangle, t Triangle) bool {
	for _, ts := range tris {
		if ts.A.Equal(geom.XY, t.A) && ts.B.Equal(geom.XY, t.B) && ts.C.Equal(geom.XY, t.C) {
			return true
		}
	}
	return false
}
