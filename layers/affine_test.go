// +build !e2e

package layers_test

import (
	"testing"

	"github.com/po3rin/gonnp/params"
	"github.com/po3rin/gonnp/layers"
	"gonum.org/v1/gonum/mat"
)

func TestAffineForward(t *testing.T) {
	type input struct {
		data   mat.Matrix
		weight mat.Matrix
		bias   mat.Vector
	}
	tests := []struct {
		name  string
		input input
		want  mat.Matrix
	}{
		{
			name: "normal",
			input: input{
				data:   mat.NewDense(2, 2, []float64{1, 1, 1, 1}),
				weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
				bias:   mat.NewVecDense(2, []float64{1, 1}),
			},
			want: mat.NewDense(2, 2, []float64{5, 7, 5, 7}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			aff := layers.InitAffineLayer(tt.input.weight, tt.input.bias)
			if got := aff.Forward(tt.input.data); !mat.EqualApprox(got, tt.want, 1e-14) {
				t.Fatalf("want = %d, got = %d", tt.want, got)
			}
		})
	}
}

func TestAffineBackword(t *testing.T) {
	type input struct {
		out    mat.Matrix
		x      mat.Matrix
		bias   mat.Vector
		weight mat.Matrix
	}
	type want struct {
		x      mat.Matrix
		bias   mat.Vector
		weight mat.Matrix
	}
	tests := []struct {
		name  string
		input input
		want  want
	}{
		{
			name: "normal",
			input: input{
				out:    mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
				x:      mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
				weight: mat.NewDense(2, 2, []float64{1, 2, 3, 4}),
			},
			want: want{
				x:      mat.NewDense(2, 2, []float64{5, 11, 11, 25}),
				bias:   mat.NewVecDense(2, []float64{4, 6}),
				weight: mat.NewDense(2, 2, []float64{10, 14, 14, 20}),
			},
		},
		{
			name: "real float v1",
			input: input{
				out: mat.NewDense(3, 3, []float64{
					1.01615691e-07, 5.54802877e-06, -3.33030421e-01,
					-3.33333296e-01, 7.50844045e-07, 1.50811058e-05,
					-3.33331292e-01, 8.23400476e-04, 3.32183461e-01,
				}),
				x:      mat.NewDense(3, 2, []float64{2, 4, 2, 3, 3, 6}),
				weight: mat.NewDense(2, 3, []float64{1, 1, 1, 1, 2, 3}),
			},
			want: want{
				x: mat.NewDense(3, 2, []float64{
					-3.33024772e-01, -9.99080066e-01,
					-3.33317464e-01, -3.33286551e-01,
					-3.24431084e-04, 6.64865891e-01,
				}),
				bias: mat.NewVecDense(3, []float64{-0.66666449, 0.0008297, -0.00083188}),
				weight: mat.NewDense(2, 3, []float64{
					-1.66666027e+00, 2.48279917e-03, 3.30519702e-01,
					-2.99998724e+00, 4.96484751e-03, 6.61024323e-01,
				}),
			},
		},

		{
			name: "real float v2",
			input: input{
				out: mat.NewDense(3, 2, []float64{
					-3.33024772e-01, -9.99080066e-01,
					-3.33317464e-01, -3.33286551e-01,
					-3.24431084e-04, 6.64865891e-01,
				}),
				x:      mat.NewDense(3, 3, []float64{1, 0, 1, 1, 1, 0, 1, 1, 1}),
				weight: mat.NewDense(3, 2, []float64{1, 1, 1, 2, 1, 3}),
			},
			want: want{
				x: mat.NewDense(3, 3, []float64{
					-1.33210484, -2.3311849, -3.33026497,
					-0.66660401, -0.99989057, -1.33317712,
					0.66454146, 1.32940735, 1.99427324,
				}),
				bias: mat.NewVecDense(2, []float64{-0.66666667, -0.66750073}),
				weight: mat.NewDense(3, 2, []float64{
					-0.66666667, -0.66750073,
					-0.3336419, 0.33157934,
					-0.3333492, -0.33421418,
				}),
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			aff := layers.InitAffineLayer(tt.input.weight, tt.input.bias)
			aff.X = tt.input.x

			got := aff.Backward(tt.input.out)
			if !mat.EqualApprox(got, tt.want.x, 1e-7) {
				t.Errorf("x:\nwant = %d\ngot = %d", tt.want.x, got)
			}
			if !mat.EqualApprox(aff.GetGrad().Weight, tt.want.weight, 1e-7) {
				t.Errorf("grad.weight:\nwant = %d\ngot = %d", tt.want.weight, aff.GetGrad().Weight)
			}
			if !mat.EqualApprox(aff.GetGrad().Bias, tt.want.bias, 1e-7) {
				t.Errorf("grad.bias\nwant = %d\ngot = %d", tt.want.bias, aff.GetGrad().Bias)
			}
		})
	}
}

func TestTimeAffineForword(t *testing.T) {
	type input struct {
		x      []mat.Matrix
		bias   mat.Vector
		weight mat.Matrix
	}
	type want struct {
		x      []mat.Matrix
		bias   mat.Vector
		weight mat.Matrix
	}
	tests := []struct {
		name  string
		input input
		want  want
	}{
		{
			name: "real float v1",
			input: input{
				x: []mat.Matrix{
					mat.NewDense(2, 3, []float64{
						-0.89664584, -0.71520257, -0.9822494,
						0.79983157, 0.917308, -0.85604465,
					}),
					mat.NewDense(2, 3, []float64{
						0.9681212, 0.8020726, 0.9271489,
						-0.70771986, -0.94187826, 0.683955,
					}),
					mat.NewDense(2, 3, []float64{
						-0.48222315, 0.3682375, 0.08181943,
						0.5935824, 0.5005101, 0.81340915,
					}),
				},
				weight: mat.NewDense(3, 7, []float64{
					-0.42660522, 0.9266366, 0.20234357, -0.27710852, 0.66819334, -0.1316577, -0.15437792,
					0.75730014, 0.660661, -0.98291314, -0.7354749, 0.49242494, -0.92058617, 0.45291832,
					0.5926396, 1.0414792, 0.27277708, -1.2767018, -0.2673152, 0.2070886, 0.56560814,
				}),
				bias: mat.NewVecDense(7, []float64{
					-0.7843269, 0.5874374, 0.09880708, -0.1970174, 0.14299354, 0.03991139, 0.11219494,
				}),
			},
			want: want{
				x: []mat.Matrix{
					mat.NewDense(2, 7, []float64{
						-1.5255561e+00, -1.7389262e+00, 3.5242343e-01, 1.8315039e+00, -5.4575264e-01, 6.1295468e-01, -6.2887931e-01,
						-9.3818778e-01, 1.0430675e+00, -8.7449557e-01, -4.0075183e-04, 1.3579748e+00, -1.0871308e+00, -8.0001622e-02,
					}),
					mat.NewDense(2, 7, []float64{
						-4.0457606e-02, 2.9800382e+00, -2.4076255e-01, -2.2388890e+00, 9.3700528e-01, -6.3392419e-01, 8.5041475e-01,
						-7.9035562e-01, 2.1701038e-02, 1.0679563e+00, -1.8138100e-01, -9.7653604e-01, 1.1418076e+00, 1.8170786e-01,
					}),
					mat.NewDense(2, 7, []float64{
						-2.5125223e-01, 4.6908516e-01, -3.3839470e-01, -4.3867770e-01, -1.9767016e-02, -2.1865070e-01, 3.9969879e-01,
						-1.7645741e-01, 2.3152888e+00, -5.1163912e-02, -1.7680976e+00, 5.6864834e-01, -3.3055320e-01, 7.0731997e-01,
					}),
				},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			aff := layers.InitTimeAffineLayer(tt.input.weight, tt.input.bias)
			got := aff.Forward(tt.input.x)

			for i, m := range got {
				if !mat.EqualApprox(m, tt.want.x[i], 1e-7) {
					t.Errorf("x:\nwant = %d\ngot = %d", m, tt.want.x[i])
				}
			}
		})
	}
}

func TestTimeAffineBackward(t *testing.T) {
	type input struct {
		x      []mat.Matrix
		dout   []mat.Matrix
		bias   mat.Vector
		weight mat.Matrix
	}
	type want struct {
		dx []mat.Matrix
		db mat.Vector
		dw mat.Matrix
	}
	tests := []struct {
		name  string
		input input
		want  want
	}{
		{
			name: "real float v1",
			input: input{
				dout: []mat.Matrix{
					mat.NewDense(2, 7, []float64{
						0.00494628, 0.01965272, 0.00899598, -0.06688424, 0.02306069, 0.00150864, 0.00871993,
						0.0195298, 0.01207395, 0.0152868, 0.01204123, -0.08304098, 0.01845012, 0.00565909,
					}),
					mat.NewDense(2, 7, []float64{
						0.00532621, -0.09632242, 0.02016217, 0.00989491, 0.01254559, 0.01734822, 0.0310453,
						0.00599629, 0.02833164, 0.02569082, 0.00212024, 0.00793114, -0.08957582, 0.01950567,
					}),
					mat.NewDense(2, 7, []float64{
						0.00230834, 0.07030352, 0.02027488, 0.01877391, 0.00359328, 0.00976891, -0.12502284,
						0.00539465, -0.11443422, 0.02818465, 0.01574317, 0.00912328, 0.02319951, 0.03278896,
					}),
				},
				x: []mat.Matrix{
					mat.NewDense(2, 3, []float64{
						0.790856, -0.77606046, 0.9863894,
						0.7994567, -0.58017415, -0.5991825,
					}),
					mat.NewDense(2, 3, []float64{
						-0.5466146, -0.6302573, -0.10507407,
						-0.99380684, 0.5723499, -0.8098436,
					}),
					mat.NewDense(2, 3, []float64{
						-0.98092574, 0.136245, 0.6034575,
						-0.69302976, 0.27515283, 0.1430181,
					}),
				},
				weight: mat.NewDense(3, 7, []float64{
					0.46211845, -1.3929424, -0.5151246, 0.44934267, 0.9243408, -0.7190074, -1.3576807,
					0.2990263, -0.6311297, 0.35085726, 0.1964794, 0.00250098, 0.60181165, -0.2400923,
					-0.5114243, 0.5369745, 0.02140709, 1.6756499, -0.49188933, -1.1934503, 0.5509024,
				}),
				bias: mat.NewVecDense(7, []float64{
					-0.6961091, 0.39457014, 0.1895368, 0.08146151, 0.22844239, -0.04171056, -0.15619123,
				}),
			},
			want: want{
				dw: mat.NewDense(3, 7, []float64{
					0.00465151, 0.06003391, -0.05663794, -0.08011147, -0.07273705, 0.06982093, 0.07498012,
					-0.01329535, 0.03275844, -0.00333621, 0.04678701, 0.02991404, -0.06636326, -0.02646467,
					-0.01007414, 0.02538655, -0.00694417, -0.06236475, 0.06823549, 0.07036576, -0.08460474,
				}),
				db: mat.NewVecDense(7, []float64{
					0.04350157, -0.0803948, 0.1185953, -0.00831078, -0.02678701, -0.0193004, -0.02730388,
				}),
				dx: []mat.Matrix{
					mat.NewDense(2, 3, []float64{
						-0.05138502, -0.02203741, -0.11219861,
						-0.10796437, 0.01548614, 0.0389448,
					}),
					mat.NewDense(2, 3, []float64{
						0.08766638, 0.0744208, -0.04720697,
						-0.00372034, -0.0652286, 0.12999824,
					}),
					mat.NewDense(2, 3, []float64{
						0.06716839, 0.00302694, -0.01383841,
						0.10168416, 0.09293015, -0.05133541,
					}),
				},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			aff := layers.InitTimeAffineLayer(tt.input.weight, tt.input.bias)
			aff.Param = params.Param{
				Weight: tt.input.weight,
				Bias:   tt.input.bias,
			}
			aff.X = tt.input.x

			got := aff.Backward(tt.input.dout)

			for i, m := range got {
				if !mat.EqualApprox(m, tt.want.dx[i], 1e-7) {
					t.Errorf("dx:\nwant = %d\ngot = %d", m, tt.want.dx[i])
				}
			}

			if !mat.EqualApprox(aff.Grad.Weight, tt.want.dw, 1e-7) {
				t.Errorf("grad.weight:\nwant = %d\ngot = %d", tt.want.dw, aff.Param.Weight)
			}
			if !mat.EqualApprox(aff.Grad.Bias, tt.want.db, 1e-7) {
				t.Errorf("grad.bias\nwant = %d\ngot = %d", tt.want.db, aff.Param.Bias)
			}
		})
	}
}
