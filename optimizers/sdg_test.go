package optimizers_test

import (
	"testing"

	"github.com/po3rin/gonlp/entity"
	"github.com/po3rin/gonlp/optimizers"
	"gonum.org/v1/gonum/mat"
)

func TestSDGUpdate(t *testing.T) {
	tests := []struct {
		name       string
		lr         float64
		params     []entity.Param
		grads      []entity.Grad
		wantParams []entity.Param
	}{
		{
			name: "lr=1",
			lr:   1,
			params: []entity.Param{
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
					Bias:   mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
				},
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{2, 3, 4, 5}),
					Bias:   mat.NewDense(2, 2, []float64{6, 7, 8, 9}),
				},
			},
			grads: []entity.Grad{
				entity.Grad{
					Weight: mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
					Bias:   mat.NewDense(2, 2, []float64{2, 2, 2, 2}),
				},
				entity.Grad{
					Weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
					Bias:   mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
				},
			},
			wantParams: []entity.Param{
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{0, 1, 2, 3}),
					Bias:   mat.NewDense(2, 2, []float64{3, 4, 5, 6}),
				},
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
					Bias:   mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
				},
			},
		},
		{
			name: "lr=0.1",
			lr:   0.1,
			params: []entity.Param{
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
					Bias:   mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
				},
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{2, 3, 4, 5}),
					Bias:   mat.NewDense(2, 2, []float64{6, 7, 8, 9}),
				},
			},
			grads: []entity.Grad{
				entity.Grad{
					Weight: mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
					Bias:   mat.NewDense(2, 2, []float64{2, 2, 2, 2}),
				},
				entity.Grad{
					Weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
					Bias:   mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
				},
			},
			wantParams: []entity.Param{
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{0.9, 1.9, 2.9, 3.9}),
					Bias:   mat.NewDense(2, 2, []float64{4.8, 5.8, 6.8, 7.8}),
				},
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{1.9, 2.8, 3.7, 4.6}),
					Bias:   mat.NewDense(2, 2, []float64{5.5, 6.4, 7.3, 8.2}),
				},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			optimizer := optimizers.InitSDG(tt.lr)
			got := optimizer.Update(tt.params, tt.grads)
			for i := 0; i < len(tt.wantParams); i++ {
				if !mat.EqualApprox(tt.wantParams[i].Weight, got[i].Weight, 1e-14) {
					t.Errorf("unexpected weight: want = %d, got = %d\n", tt.wantParams[i].Weight, got[i].Weight)
				}
				if !mat.EqualApprox(tt.wantParams[i].Bias, got[i].Bias, 1e-14) {
					t.Errorf("unexpected bias: want = %d, got = %d\n", tt.wantParams[i].Bias, got[i].Bias)
				}
			}
		})
	}
}
