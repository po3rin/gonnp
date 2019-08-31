package layers_test

import (
	"testing"

	"github.com/po3rin/gonnp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestRNNForward(t *testing.T) {
	tests := []struct {
		name      string
		wx        mat.Matrix
		wh        mat.Matrix
		b         mat.Vector
		x         mat.Matrix
		hPrev     mat.Matrix
		wantHNext mat.Matrix
	}{
		{
			name: "real float",
			wx: mat.NewDense(4, 2, []float64{
				-0.33697683, -0.02020934,
				0.02767114, 0.47382835,
				0.4035237, 0.762225,
				0.45493788, 0.878702,
			}),
			wh: mat.NewDense(2, 2, []float64{
				0.3348753, 0.81213206,
				-0.8696501, -0.5360801,
			}),
			b: mat.NewVecDense(2, []float64{
				-0.11941066, -0.11335015,
			}),
			x: mat.NewDense(3, 4, []float64{
				-0.00816947, 0.0033809, 0.00620988, 0.00897177,
				0.00201355, -0.00481653, -0.01092697, -0.00571691,
				-0.00814809, 0.00379368, 0.00155447, -0.00541426,
			}),
			hPrev: mat.NewDense(3, 2, []float64{
				-0.01923645, -0.10349256,
				-0.04790352, -0.14870238,
				0.01784713, -0.13251244,
			}),
			wantHNext: mat.NewDense(3, 2, []float64{
				-0.0264101, -0.05903973,
				-0.01395434, -0.08798482,
				0.00282016, -0.0294206,
			}),
		},
	}

	for _, tt := range tests {
		r := layers.InitRNNLayer(tt.wx, tt.wh, tt.b)
		got := r.Forward(tt.x, tt.hPrev)
		if !mat.EqualApprox(got, tt.wantHNext, 1e-7) {
			t.Fatalf("want = %v, got = %v", tt.wantHNext, got)
		}
	}
}

func TestRNNBackward(t *testing.T) {
	tests := []struct {
		name   string
		wx     mat.Matrix
		wh     mat.Matrix
		x      mat.Matrix
		hPrev  mat.Matrix
		hNext  mat.Matrix
		dhNext mat.Matrix

		wantDwx    mat.Matrix
		wantDwh    mat.Matrix
		wantDb     mat.Vector
		wantDx     mat.Matrix
		wantDhPrev mat.Matrix
	}{
		{
			name: "real float",
			wx: mat.NewDense(4, 2, []float64{
				-0.25771597, 0.20832862,
				1.3869158, 0.01348229,
				0.0574846, 1.0725518,
				0.15927838, -0.4052679,
			}),
			wh: mat.NewDense(2, 2, []float64{
				0.6486265, -0.46730414,
				0.48135573, 0.73577607,
			}),
			x: mat.NewDense(3, 4, []float64{
				0.01683594, -0.04555828, -0.03624132, 0.00278626,
				0.0039673, -0.05200959, 0.00588301, 0.00182244,
				0.0152199, -0.00908084, 0.02983364, -0.00824076,
			}),
			hPrev: mat.NewDense(3, 2, []float64{
				-0.21135336, 0.08441298,
				-0.19091837, 0.02665257,
				-0.21588996, -0.06760632,
			}),
			hNext: mat.NewDense(3, 2, []float64{
				-0.33200756, 0.00493498,
				-0.3478489, -0.00430981,
				-0.3523735, -0.02929596,
			}),
			dhNext: mat.NewDense(3, 2, []float64{
				0.13687567, 0.10796446,
				-0.06647974, -0.05776975,
				0.0489943, 0.06153633,
			}),
			wantDwx: mat.NewDense(4, 2, []float64{
				0.00247168, 0.00252423,
				-0.0028989, -0.00247235,
				-0.00347735, -0.00241826,
				-0.00012078, -0.00031114,
			}),
			wantDwh: mat.NewDense(2, 2, []float64{
				-0.02384786, -0.02506267,
				0.00582198, 0.00341702,
			}),
			wantDb: mat.NewVecDense(2, []float64{
				0.10626306, 0.11167667,
			}),
			wantDx: mat.NewDense(3, 4, []float64{
				-0.00889518, 0.17036527, 0.12279559, -0.02435527,
				0.00302496, -0.08182435, -0.06531905, 0.01410423,
				0.00174997, 0.06034263, 0.06841097, -0.01808253,
			}),
			wantDhPrev: mat.NewDense(3, 2, []float64{
				0.02854392, 0.13805908,
				-0.01090745, -0.0706332,
				-0.00089841, 0.06589347,
			}),
		},
	}

	for _, tt := range tests {
		r := layers.InitRNNLayer(tt.wx, tt.wh, nil)

		r.SetRNNCacheX(tt.x)
		r.SetRNNCacheHPrev(tt.hPrev)
		r.SetRNNCacheNextPrev(tt.hNext)

		gotDx, gotDhPrev := r.Backward(tt.dhNext)
		if !mat.EqualApprox(gotDx, tt.wantDx, 1e-7) {
			t.Errorf("want = %v, got = %v", tt.wantDx, gotDx)
		}
		if !mat.EqualApprox(gotDhPrev, tt.wantDhPrev, 1e-7) {
			t.Errorf("want = %v, got = %v", tt.wantDhPrev, gotDhPrev)
		}

		if !mat.EqualApprox(r.Grad.Weight, tt.wantDwx, 1e-7) {
			t.Errorf("want = %v, got = %v", tt.wantDwx, r.Grad.Weight)
		}
		if !mat.EqualApprox(r.Grad.WeightH, tt.wantDwh, 1e-7) {
			t.Errorf("want = %v, got = %v", tt.wantDwh, r.Grad.WeightH)
		}
		if !mat.EqualApprox(r.Grad.Bias, tt.wantDb, 1e-7) {
			t.Errorf("want = %v, got = %v", tt.wantDb, r.Grad.Bias)
		}
	}
}
