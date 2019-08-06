// +build !e2e

package optimizers_test

import (
	"testing"

	"github.com/po3rin/gonlp/entity"
	"github.com/po3rin/gonlp/optimizers"
	"gonum.org/v1/gonum/mat"
)

func TestAdamUpdate(t *testing.T) {

	tests := []struct {
		name       string
		lr         float64
		beta1      float64
		beta2      float64
		params     []entity.Param
		grads      []entity.Grad
		wantParams []entity.Param
	}{
		{
			name:  "lr=1",
			lr:    0.001,
			beta1: 0.9,
			beta2: 0.999,
			params: []entity.Param{
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
					Bias:   mat.NewVecDense(4, []float64{5, 6, 7, 8}),
				},
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{2, 3, 4, 5}),
					Bias:   mat.NewVecDense(4, []float64{6, 7, 8, 9}),
				},
			},
			grads: []entity.Grad{
				entity.Grad{
					Weight: mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
					Bias:   mat.NewVecDense(4, []float64{2, 2, 2, 2}),
				},
				entity.Grad{
					Weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
					Bias:   mat.NewVecDense(4, []float64{5, 6, 7, 8}),
				},
			},
			wantParams: []entity.Param{
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{0, 1, 2, 3}),
					Bias:   mat.NewVecDense(4, []float64{3, 4, 5, 6}),
				},
				entity.Param{
					Weight: mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
					Bias:   mat.NewVecDense(4, []float64{1, 1, 1, 1}),
				},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			optimizer := optimizers.InitAdam(tt.lr, tt.beta1, tt.beta2)
			_ = optimizer.Update(tt.params, tt.grads)
			// for i := 0; i < len(tt.wantParams); i++ {
			// 	if !mat.EqualApprox(tt.wantParams[i].Weight, got[i].Weight, 1e-14) {
			// 		t.Errorf("unexpected weight: want = %d, got = %d\n", tt.wantParams[i].Weight, got[i].Weight)
			// 	}
			// 	if !mat.EqualApprox(tt.wantParams[i].Bias, got[i].Bias, 1e-14) {
			// 		t.Errorf("unexpected bias: want = %d, got = %d\n", tt.wantParams[i].Bias, got[i].Bias)
			// 	}
			// }
		})
	}
}
