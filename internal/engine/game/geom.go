package game

import (
	"math"

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

// NormalizeTriangle ensures that the 3 points of a triangle are always stored with smaller point first and greater point last
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

// GenerateTrianglesFromLines generates all possible unique triangles from a given array of lines
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

					// get vertices
					verticeA := GetCommonPoint(uniqueLines[i], uniqueLines[j])
					verticeB := GetCommonPoint(uniqueLines[j], uniqueLines[k])
					verticeC := GetCommonPoint(uniqueLines[k], uniqueLines[i])

					// only if each point is different
					if !verticeA.Equal(geom.XY, verticeB) && !verticeB.Equal(geom.XY, verticeC) && !verticeA.Equal(geom.XY, verticeC) {
						triangle := NormalizeTriangle(verticeA, verticeB, verticeC)
						// only add new triangles
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

// GetCommonPoint get common endpoint of a line
func GetCommonPoint(line1, line2 Line) geom.Coord {
	if line1.A.Equal(geom.XY, *line2.A) || line1.B.Equal(geom.XY, *line2.A) {
		return *line2.A
	}
	return *line2.B
}

// Checks if a triangle already exists in a list of triangles
func Contains(tris []Triangle, t Triangle) bool {
	for _, ts := range tris {
		if ts.A.Equal(geom.XY, t.A) && ts.B.Equal(geom.XY, t.B) && ts.C.Equal(geom.XY, t.C) {
			return true
		}
	}
	return false
}

// Función para calcular la orientación de tres puntos en 2D
func orient2d(a, b, c []int) int {
	return (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])
}

// Función para añadir sesgo en el cálculo
func bias(p0, p1 []int) int {
	if (p0[1] == p1[1] && p0[0] > p1[0]) || p0[1] > p1[1] {
		return 0
	}
	return -1
}

// func to generate points inside a triangle
func renderTriangle(t Triangle) []geom.Coord {
	v0 := []int{int(t.A.X()), int(t.A.Y())}
	v1 := []int{int(t.B.X()), int(t.B.Y())}
	v2 := []int{int(t.C.X()), int(t.C.Y())}

	// make sure all points are oriented in anticlockwise direction
	if orient2d(v0, v1, v2) < 0 {
		v0, v1 = v1, v0
	}

	// get limits of the triangle area
	x0 := int(math.Min(float64(v0[0]), math.Min(float64(v1[0]), float64(v2[0]))))
	x1 := int(math.Max(float64(v0[0]), math.Max(float64(v1[0]), float64(v2[0]))))
	y0 := int(math.Min(float64(v0[1]), math.Min(float64(v1[1]), float64(v2[1]))))
	y1 := int(math.Max(float64(v0[1]), math.Max(float64(v1[1]), float64(v2[1]))))

	var pointsInTriangle []geom.Coord

	// Recorrer el área delimitada por el triángulo
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			p := []int{x, y}

			// Cálculo de las coordenadas baricéntricas
			w0 := orient2d(v1, v2, p) + bias(v1, v2)
			w1 := orient2d(v2, v0, p) + bias(v2, v0)
			w2 := orient2d(v0, v1, p) + bias(v0, v1)

			// Si los tres valores son mayores o iguales a 0, el punto está dentro del triángulo
			if w0 >= 0 && w1 >= 0 && w2 >= 0 {
				pos := geom.Coord{float64(p[0]), float64(p[1])}
				pointsInTriangle = append(pointsInTriangle, pos)
			}
		}
	}

	return pointsInTriangle
}

func pointIsInLine(p, lineEndpoint1, lineEndpoint2 geom.Coord) bool {
	return (p[1]-lineEndpoint1[1])*(lineEndpoint2[0]-lineEndpoint1[0]) == (lineEndpoint2[1]-lineEndpoint1[1])*(p[0]-lineEndpoint1[0])
}

func linesIntersect(line1Start, line1End, line2Start, line2End geom.Coord) bool {
	// Unpack the coordinates for easy access
	x1, y1 := line1Start.X(), line1Start.Y()
	x2, y2 := line1End.X(), line1End.Y()
	x3, y3 := line2Start.X(), line2Start.Y()
	x4, y4 := line2End.X(), line2End.Y()

	// Calculate the direction of the lines
	denominator := (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
	if denominator == 0 {
		return false // Lines are parallel or coincident
	}

	// Calculate the intersection point using the determinant
	t := ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denominator
	u := ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / denominator

	// If 0 <= t <= 1 and 0 <= u <= 1, there is an intersection
	return t >= 0 && t <= 1 && u >= 0 && u <= 1
}
