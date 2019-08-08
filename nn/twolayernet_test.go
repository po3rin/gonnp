// +build !e2e

package nn_test

// func getOutFromLayer(tb *testing.T, l layers.Layer) mat.Matrix {
// 	tb.Helper()
// 	if a, ok := l.(*layers.Affine); ok {
// 		return a.X
// 	}
// 	if s, ok := l.(*layers.Sigmoid); ok {
// 		return s.X
// 	}
// 	if r, ok := l.(*layers.Relu); ok {
// 		return r.X
// 	}
// 	return nil
// }

// func getOutFromLossLayer(tb *testing.T, l layers.LossLayer) mat.Matrix {
// 	tb.Helper()
// 	if sl, ok := l.(*layers.SoftmaxWithLoss); ok {
// 		return sl.X
// 	}
// 	return nil
// }

// func TestTwoLayer(t *testing.T) {
// 	tests := []struct {
// 		name         string
// 		inputSize    int
// 		hiddenSize   int
// 		outputSize   int
// 		data         mat.Matrix
// 		teacher      mat.Matrix
// 		beforeParams []entity.Param
// 		afterParams  []entity.Param
// 		afterGrads   []entity.Grad
// 		loss         float64
// 		outs         []mat.Matrix
// 		lossOut      mat.Matrix
// 		params       []entity.Param
// 		grads        []entity.Grad
// 	}{
// 		// want data is calicurated by colab.
// 		// https://colab.research.google.com/drive/1XJk9_YmqliNP9nDJWsVatcZIZDSEriC7
// 		{
// 			name:       "3:2:3",
// 			inputSize:  3,
// 			hiddenSize: 2,
// 			outputSize: 3,
// 			data:       mat.NewDense(3, 3, []float64{1, 0, 1, 1, 1, 0, 1, 1, 1}),
// 			teacher:    mat.NewDense(3, 3, []float64{0, 0, 1, 1, 0, 0, 1, 0, 0}),
// 			beforeParams: []entity.Param{
// 				entity.Param{
// 					Weight: mat.NewDense(3, 2, []float64{1, 1, 1, 2, 1, 3}),
// 				},
// 				entity.Param{
// 					Weight: mat.NewDense(2, 3, []float64{1, 1, 1, 1, 2, 3}),
// 				},
// 			},
// 			afterGrads: []entity.Grad{
// 				entity.Grad{
// 					Weight: mat.NewDense(3, 2, []float64{
// 						-0.66666667, -0.66750073,
// 						-0.3336419, 0.33157934,
// 						-0.3333492, -0.33421418,
// 					}),
// 					Bias: mat.NewVecDense(2, []float64{
// 						-0.66666667, -0.66750073,
// 					}),
// 				},
// 				entity.Grad{
// 					Weight: mat.NewDense(2, 3, []float64{
// 						-1.66666027e+00, 2.48279917e-03, 3.30519702e-01,
// 						-2.99998724e+00, 4.96484751e-03, 6.61024323e-01,
// 					}),
// 					Bias: mat.NewVecDense(3, []float64{
// 						-0.66666449, 0.0008297, -0.00083188,
// 					}),
// 				},
// 			},
// 			afterParams: []entity.Param{
// 				entity.Param{
// 					Weight: mat.NewDense(3, 2, []float64{
// 						1.00666667, 1.00667501,
// 						1.00333642, 1.99668421,
// 						1.00333349, 3.00334214,
// 					}),
// 					Bias: mat.NewVecDense(2, []float64{
// 						0.00666667, 0.00667501,
// 					}),
// 				},
// 				entity.Param{
// 					Weight: mat.NewDense(2, 3, []float64{
// 						1.0166666, 0.99997517, 0.9966948,
// 						1.02999987, 1.99995035, 2.99338976,
// 					}),
// 					Bias: mat.NewVecDense(3, []float64{
// 						6.66664487e-03, -8.29699349e-06, 8.31879353e-06,
// 					}),
// 				},
// 			},
// 			loss: 11.452195926332474,
// 			outs: []mat.Matrix{
// 				mat.NewDense(3, 3, []float64{1, 0, 1, 1, 1, 0, 1, 1, 1}),
// 				mat.NewDense(3, 2, []float64{2, 4, 2, 3, 3, 6}),
// 				mat.NewDense(3, 2, []float64{2, 4, 2, 3, 3, 6}),
// 			},
// 			lossOut: mat.NewDense(3, 3, []float64{
// 				3.04847074443256e-07, 1.6644086307618e-05, 0.00090873632138,
// 				1.12146971388934e-07, 2.2525321346561e-06, 4.5243317361302e-05,
// 				6.12301716965576e-06, 0.002470201429289, 0.9965503823023,
// 			}),
// 		},
// 	}

// 	// sets generator to init constant value for test.
// 	weightGenerator := func(r, c int) mat.Matrix {
// 		a := make([]float64, 0, r*c)
// 		for i := 0; i < r; i++ {
// 			for j := 0; j < c; j++ {
// 				a = append(a, float64(i*j+1))
// 			}
// 		}
// 		return mat.NewDense(r, c, a)
// 	}
// 	defer nn.UseCustomWightGenerator(weightGenerator)()

// 	biasGenerator := mat.NewVecDense
// 	defer nn.UseCustomBiasGenerator(biasGenerator)()

// 	for _, tt := range tests {
// 		tt := tt
// 		t.Run(tt.name, func(t *testing.T) {
// 			// inits
// 			nn := nn.NewTwoLayerNet(tt.inputSize, tt.hiddenSize, tt.outputSize)

// 			// checks before params
// 			params := nn.GetParams()
// 			for i := 0; i < len(params); i++ {
// 				if !mat.EqualApprox(params[i].Weight, tt.beforeParams[i].Weight, 1e-14) {
// 					t.Fatalf("want = %v, got = %v", tt.beforeParams[i].Weight, params[i].Weight)
// 				}
// 			}

// 			// forward
// 			loss := nn.Forward(tt.teacher, tt.data)

// 			// assert loss
// 			if loss != tt.loss {
// 				t.Fatalf("want = %v, got = %v", tt.loss, loss)
// 			}

// 			// assert outs
// 			outs := make([]mat.Matrix, 0, 3)
// 			for _, l := range nn.Layers {
// 				out := getOutFromLayer(t, l)
// 				if out == nil {
// 					t.Fatal("failed to get layer output")
// 				}
// 				outs = append(outs, out)
// 			}

// 			if len(outs) != len(tt.outs) {
// 				t.Fatalf("unexpected len of outputs. want: %v, got: %v\n", len(tt.outs), len(outs))
// 			}
// 			for i, o := range outs {
// 				if !mat.EqualApprox(tt.outs[i], o, 1e-14) {
// 					t.Errorf("want = %v, got = %v\n", tt.outs[i], o)
// 				}
// 			}

// 			y := getOutFromLossLayer(t, nn.LossLayer)
// 			if !mat.EqualApprox(tt.lossOut, y, 1e-14) {
// 				t.Errorf("want = %v, got = %v\n", tt.lossOut, y)
// 			}

// 			// backward
// 			nn.Backward()

// 			// assert grads
// 			grads := nn.GetGrads()
// 			for i := 0; i < len(grads); i++ {
// 				if !mat.EqualApprox(grads[i].Weight, tt.afterGrads[i].Weight, 1e-6) {
// 					t.Errorf("grad.Weight\nwant = %v\ngot = %v", tt.afterGrads[i].Weight, grads[i].Weight)
// 				}
// 				if !mat.EqualApprox(grads[i].Bias, tt.afterGrads[i].Bias, 1e-6) {
// 					t.Errorf("grad.Bias\nwant = %v\ngot = %v", tt.afterGrads[i].Bias, grads[i].Bias)
// 				}
// 			}

// 			// update params
// 			optimizer := optimizers.InitSDG(0.01)
// 			nn.UpdateParams(
// 				optimizer.Update(
// 					nn.GetParams(), nn.GetGrads(),
// 				),
// 			)

// 			// checks after params
// 			params = nn.GetParams()
// 			for i := 0; i < len(params); i++ {
// 				if !mat.EqualApprox(params[i].Weight, tt.afterParams[i].Weight, 1e-7) {
// 					t.Errorf("after param.Weight\nwant = %v\ngot = %v\n", tt.afterParams[i].Weight, params[i].Weight)
// 				}
// 				if !mat.EqualApprox(params[i].Bias, tt.afterParams[i].Bias, 1e-7) {
// 					t.Errorf("after param.Bias\nwant = %v\ngot = %v\n", tt.afterParams[i].Bias, params[i].Bias)
// 				}
// 			}
// 		})
// 	}
// }
