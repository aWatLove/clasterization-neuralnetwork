package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
)

const (
	MapWidth       = 60
	MapHeight      = 60
	InputDimension = 9
	NumIterations  = 10000
	LearningRate   = 0.1
)

type SOM struct {
	Map [][]*Neuron
}

type Neuron struct {
	Weights []float64
}

func NewSOM() *SOM {
	som := &SOM{}
	som.Initialize()
	return som
}

func (som *SOM) Initialize() {
	som.Map = make([][]*Neuron, MapWidth)
	for i := range som.Map {
		som.Map[i] = make([]*Neuron, MapHeight)
		for j := range som.Map[i] {
			som.Map[i][j] = &Neuron{Weights: make([]float64, InputDimension)}
			for k := range som.Map[i][j].Weights {
				som.Map[i][j].Weights[k] = rand.Float64()
			}
		}
	}
}

func (som *SOM) FindWinner(input []float64) (int, int) {
	minDist := math.Inf(1)
	var winnerX, winnerY int
	for x, row := range som.Map {
		for y, neuron := range row {
			dist := euclideanDistance(input, neuron.Weights)
			if dist < minDist {
				minDist = dist
				winnerX = x
				winnerY = y
			}
		}
	}
	return winnerX, winnerY
}

func euclideanDistance(vec1, vec2 []float64) float64 {
	sum := 0.0
	for i := range vec1 {
		sum += math.Pow(vec1[i]-vec2[i], 2)
	}
	return math.Sqrt(sum)
}

func (som *SOM) Train(data [][]float64, numIterations int) {
	for i := 0; i < numIterations; i++ {
		input := data[rand.Intn(len(data))]
		winnerX, winnerY := som.FindWinner(input)
		radius := getRadius(MapWidth/2, i, float64(NumIterations)/5)

		for x, row := range som.Map {
			for y, neuron := range row {
				distance := math.Sqrt(math.Pow(float64(x-winnerX), 2) + math.Pow(float64(y-winnerY), 2))
				neighborhood := math.Exp(-math.Pow(distance, 2) / (2 * math.Pow(radius, 2)))

				for k := range neuron.Weights {
					neuron.Weights[k] += LearningRate * neighborhood * (input[k] - neuron.Weights[k])
				}
			}
		}
	}
}

func getRadius(initialRadius float64, iteration int, timeConstant float64) float64 {
	return initialRadius * math.Exp(-float64(iteration)/timeConstant)
}

func main() {
	som := NewSOM()

	// Input data
	input := [][]float64{
		{0, 0, 0.531971070780392, 0.2176470588235294, 0.41379310344827586, 0.31638418079096043, 0.4507936507936508, 0.0, 0.0},
		{1, 0, 0.5779751671500707, 0.3117647058823529, 0.7655172413793104, 1.0, 1.0, 0.0, 0.0},
		{1, 1, 0.10772094395280236, 0.2235294117647059, 0.18689655172413793, 0.06438395185544625, 0.03714285714285714, 0.0, 0.0},
		{1, 0, 0.09023032904148731, 0.16647058823529413, 0.21379310344827586, 0.04066444511781445, 0.0380952380952381, 0.0, 0.0},
		{1, 1, 0.33905112176380305, 0.15294117647058825, 0.6896551724137931, 0.2927386710322725, 0.023809523809523808, 0.0, 0.0},
		{0, 0, 0.004308390022675074, 0.0, 0.017241379310344827, 0.029510961214165567, 0.0, 0.25, 0.5},
		{1, 0, 0.04308390022675056, 0.18823529411764706, 0.06896551724137931, 0.0033321113836020386, 0.29523809523809524, 1.0, 1.0},
		{1, 1, 0.00030169895687253945, 0.09411764705882353, 0.0, 0.0, 0.0, 0.25, 0.5},
		{1, 1, 0.009501677173207795, 0.03529411764705882, 0.034482758620689655, 0.006204263532199312, 0.07142857142857142, 0.25, 0.5},
		{0, 0, 0.030508305647840533, 0.07058823529411765, 0.034482758620689655, 0.05442519694230061, 0.0, 0.25, 0.5},
		{1, 0, 0.06847440895137286, 0.15294117647058825, 0.15517241379310345, 0.17585085078544423, 0.21904761904761905, 0.0, 0.0},
		{1, 0, 0.01810802942640208, 0.24176470588235294, 0.039482758620689655, 0.07006527135425466, 0.08928571428571428, 0.125, 0.16666666666666666},
		{0, 0, 0.030508305647840533, 0.4117647058823529, 0.034482758620689655, 0.05471152866470483, 0.1523809523809524, 0.125, 0.16666666666666666},
		{0, 0, 0.04584641638225256, 0.29411764705882354, 0.20689655172413793, 0.08994315983958141, 0.10476190476190476, 0.125, 0.16666666666666666},
		{1, 1, 0.02090104279749478, 0.047058823529411764, 0.05172413793103448, 0.006141226818431377, 0.06666666666666667, 0.125, 0.16666666666666666},
		{0, 0, 1.0, 0.29411764705882354, 1.0, 1.0, 1.0, 0.0, 0.0},
		{1, 1, 0.3218029350104822, 0.4117647058823529, 0.7241379310344828, 0.5440913405626895, 0.4238095238095238, 0.0, 0.0},
		{1, 1, 0.11895114614343714, 0.3411764705882353, 0.29310344827586204, 0.31638418079096043, 0.4507936507936508, 0.0, 0.0},
		{1, 1, 0.05933882256169278, 0.15294117647058825, 0.15517241379310345, 0.6187935656836461, 0.17142857142857143, 0.0, 0.0},
		{0, 0, 0.04189678500771164, 0.18823529411764706, 0.15517241379310345, 0.0020852798777441924, 0.11428571428571428, 0.125, 0.16666666666666666},
		{1, 1, 0.029299572039024464, 1.0, 0.06896551724137931, 0.10911223504274142, 0.16666666666666666, 0.125, 0.16666666666666666},
		{0, 0, 0.0254237382075247, 0.15294117647058825, 0.034482758620689655, 0.13323859530690042, 0.03571428571428571, 0.125, 0.16666666666666666},
		{1, 0, 0.1659793510324484, 0.11764705882352941, 0.13793103448275862, 0.04889419832848591, 1.0, 0.0, 0.3333333333333333},
		{1, 0, 0.45145666286644924, 0.038823529411764705, 0.6896551724137931, 0.6141080867525648, 0.42857142857142855, 0.125, 0.16666666666666666},
		{0, 0, 0.1590098522167488, 0.213529411764705882, 0.15517241379310345, 0.0, 0.11428571428571428, 0.375, 0.0},
	}

	som.Train(input, NumIterations)

	image := visualizeSOM(som)
	saveImage(image, "som.png")
}

func visualizeSOM(som *SOM) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, MapWidth*5, MapHeight*5))

	for x := 0; x < MapWidth; x++ {
		for y := 0; y < MapHeight; y++ {
			color := som.Map[x][y].ToColor()
			for i := 0; i < 5; i++ {
				for j := 0; j < 5; j++ {
					img.Set(x*5+i, y*5+j, color)
				}
			}
		}
	}

	return img
}

func (neuron *Neuron) ToColor() color.RGBA {
	r := uint8(neuron.Weights[0] * 255)
	g := uint8(neuron.Weights[1] * 255)
	b := uint8(neuron.Weights[2] * 255)
	return color.RGBA{R: r, G: g, B: b, A: 255}
}

func saveImage(img image.Image, filename string) {
	f, err := os.Create(filename)
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	defer f.Close()
	png.Encode(f, img)
	fmt.Println("Image saved to", filename)
}
