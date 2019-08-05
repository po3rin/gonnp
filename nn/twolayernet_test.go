// +build !e2e

package nn_test

import (
	"testing"

	"github.com/po3rin/gonlp/entity"
	"github.com/po3rin/gonlp/nn"
	"gonum.org/v1/gonum/mat"
)

func TestNewTwoLayerForward(t *testing.T) {
	tests := []struct {
		name         string
		inputSize    int
		hiddenSize   int
		outputSize   int
		data         mat.Matrix
		teacher      mat.Matrix
		beforeParams []entity.Param
		loss         float64
	}{
		{
			name:       "3:2:3",
			inputSize:  3,
			hiddenSize: 2,
			outputSize: 3,
			data:       mat.NewDense(3, 3, []float64{1, 0, 1, 1, 1, 0, 1, 1, 1}),
			teacher:    mat.NewDense(3, 3, []float64{0, 0, 1, 1, 0, 0, 1, 0, 0}),
			beforeParams: []entity.Param{
				entity.Param{
					Weight: mat.NewDense(3, 2, []float64{1, 1, 1, 2, 1, 3}),
				},
				entity.Param{
					Weight: mat.NewDense(2, 3, []float64{1, 1, 1, 1, 2, 3}),
				},
			},
			loss: 11.452195926332474, //calicurated by colab.
		},
	}

	// sets weight generator to init constant value for test.
	weightGenerator := func(r, c int) mat.Matrix {
		a := make([]float64, 0, r*c)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
				a = append(a, float64(i*j+1))
			}
		}
		return mat.NewDense(r, c, a)
	}
	defer nn.UseCustomWightGenerator(weightGenerator)()

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			// inits
			nn := nn.NewTwoLayerNet(tt.inputSize, tt.hiddenSize, tt.outputSize)
			if nn.NodeSize.Input != tt.inputSize ||
				nn.NodeSize.Hidden != tt.hiddenSize ||
				nn.NodeSize.Output != tt.outputSize {
				t.Errorf("unexpected node size")
			}

			// checks before params
			params := nn.GetParams()
			for i := 0; i < len(params); i++ {
				if !mat.EqualApprox(params[i].Weight, tt.beforeParams[i].Weight, 1e-14) {
					t.Fatalf("want = %v, got = %v", tt.beforeParams[i].Weight, params[i].Weight)
				}
			}

			// forward
			loss := nn.Forward(tt.data, tt.teacher)
			if loss != tt.loss {
				t.Fatalf("want = %v, got = %v", tt.loss, loss)
			}
		})
	}
}
