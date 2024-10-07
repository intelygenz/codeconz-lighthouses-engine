package game

import (
	"math"
)

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
func renderTriangle(v0, v1, v2 []int) [][]int {

	// make sure all points are oriented in anticlockwise direction
	if orient2d(v0, v1, v2) < 0 {
		v0, v1 = v1, v0
	}

	// get limits of the triangle area
	x0 := int(math.Min(float64(v0[0]), math.Min(float64(v1[0]), float64(v2[0]))))
	x1 := int(math.Max(float64(v0[0]), math.Max(float64(v1[0]), float64(v2[0]))))
	y0 := int(math.Min(float64(v0[1]), math.Min(float64(v1[1]), float64(v2[1]))))
	y1 := int(math.Max(float64(v0[1]), math.Max(float64(v1[1]), float64(v2[1]))))

	var pointsInTriangle [][]int

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
				pointsInTriangle = append(pointsInTriangle, p)
			}
		}
	}

	return pointsInTriangle
}
