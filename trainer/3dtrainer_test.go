// +build !e2e

package trainer_test

import (
	"testing"

	"github.com/po3rin/gonnp/params"
	"github.com/po3rin/gonnp/matutil"
	"github.com/po3rin/gonnp/models"
	"github.com/po3rin/gonnp/optimizers"
	"github.com/po3rin/gonnp/trainer"
	"github.com/po3rin/gonnp/word"
	"gonum.org/v1/gonum/mat"
)

func TestRmDuplicate(t *testing.T) {
	tests := []struct {
		name       string
		params     []params.Param
		grads      []params.Grad
		wantParams []params.Param
		wantGrads  []params.Grad
	}{
		{
			name: "real value",
			params: []params.Param{
				params.Param{
					Weight: mat.NewDense(7, 5, []float64{
						0.00071247, -0.00751074, 0.0051788, -0.00163169, 0.01683916,
						-0.00305428, -0.01006934, 0.00135434, 0.01265921, -0.00022962,
						-0.00295395, 0.00068649, 0.01354565, 0.01561498, 0.00121486,
						-0.0149809, 0.00873301, -0.00174364, -0.00547919, -0.00257115,
						-0.00515935, 0.0008051, -0.00196354, -0.00097649, -0.00392523,
						0.00740807, -0.01184491, 0.0029069, 0.00408089, -0.00566627,
						0.00625184, 0.00085856, 0.0048198, 0.00350936, -0.00547352,
					}),
				},
				params.Param{
					Weight: mat.NewDense(7, 5, []float64{
						0.00071247, -0.00751074, 0.0051788, -0.00163169, 0.01683916,
						-0.00305428, -0.01006934, 0.00135434, 0.01265921, -0.00022962,
						-0.00295395, 0.00068649, 0.01354565, 0.01561498, 0.00121486,
						-0.0149809, 0.00873301, -0.00174364, -0.00547919, -0.00257115,
						-0.00515935, 0.0008051, -0.00196354, -0.00097649, -0.00392523,
						0.00740807, -0.01184491, 0.0029069, 0.00408089, -0.00566627,
						0.00625184, 0.00085856, 0.0048198, 0.00350936, -0.00547352,
					}),
				},
				params.Param{
					Weight: mat.NewDense(5, 7, []float64{
						-1.18317632e-02, -2.08208330e-05, -1.87874207e-02, 1.16085364e-02, -2.41277339e-03, -4.38692343e-03, 1.33268259e-02,
						-3.21541505e-03, -2.33280253e-03, -8.06818734e-03, -1.03148615e-02, 1.07880447e-02, 1.84751983e-02, 3.45306110e-04,
						-3.47393639e-03, 1.06358228e-02, -9.85665846e-03, 6.64017297e-03, -9.66990591e-03, 1.40635937e-02, 9.53063964e-04,
						1.16737014e-02, -1.74852007e-03, 1.57040551e-02, 2.27548392e-03, 1.76936817e-02, -1.81000849e-03, -2.34262812e-03,
						-1.32313857e-03, 1.06389083e-02, 3.60980145e-03, 1.38926957e-02, -1.69368050e-02, 4.32827083e-03, 2.26744286e-02,
					}),
				},
			},
			grads: []params.Grad{
				params.Grad{
					Weight: mat.NewDense(7, 5, []float64{
						-0.00021922, 0.00025485, -0.00131402, 0.00151378, -0.00061075,
						0.00259923, 0.00153943, 0.00176951, -0.0017144, 0.00022873,
						-0.00246649, 0.00191711, -0.00064852, 0.00084333, -0.00148598,
						0., 0., 0., 0., 0.,
						0., 0., 0., 0., 0.,
						0., 0., 0., 0., 0.,
						0., 0., 0., 0., 0.,
					}),
				},
				params.Grad{
					Weight: mat.NewDense(7, 5, []float64{
						0., 0., 0., 0., 0.,
						0., 0., 0., 0., 0.,
						-0.00021922, 0.00025485, -0.00131402, 0.00151378, -0.00061075,
						0.00259923, 0.00153943, 0.00176951, -0.0017144, 0.00022873,
						-0.00246649, 0.00191711, -0.00064852, 0.00084333, -0.00148598,
						0., 0., 0., 0., 0.,
						0., 0., 0., 0., 0.,
					}),
				},
				params.Grad{
					Weight: mat.NewDense(5, 7, []float64{
						-7.23263197e-04, -1.88345180e-05, 1.94964953e-03, 9.62016851e-04, -7.23232408e-04, -7.23225987e-04, -7.23110269e-04,
						-1.64165643e-05, 7.89892226e-04, -1.26347251e-04, -5.97862361e-04, -1.63935278e-05, -1.64230318e-05, -1.64494897e-05,
						6.64884889e-04, -2.12321939e-03, 3.97048203e-04, -9.33246055e-04, 6.64817737e-04, 6.64879019e-04, 6.64835598e-04,
						8.99976459e-04, -1.76340025e-03, 3.60499897e-05, -1.87235652e-03, 8.99913672e-04, 8.99956947e-04, 8.99859699e-04,
						2.50602983e-04, -2.42614640e-03, 3.88623330e-04, 1.03511475e-03, 2.50530315e-04, 2.50615260e-04, 2.50659754e-04,
					}),
				},
			},
			wantParams: []params.Param{
				params.Param{
					Weight: mat.NewDense(7, 5, []float64{
						-2.7330520e-04, -6.5229959e-03, 4.1811997e-03, -6.3376984e-04, 1.5844310e-02,
						-2.0554923e-03, -9.0713920e-03, 2.3525597e-03, 1.1661050e-02, 7.5674366e-04,
						-3.9527710e-03, 1.6850383e-03, 1.2547255e-02, 1.6613640e-02, 2.1636263e-04,
						-1.3982113e-02, 9.7309556e-03, -7.4542401e-04, -6.4773518e-03, -1.5847827e-03,
						-6.1580711e-03, 1.8034547e-03, -2.9586919e-03, 1.9773808e-05, -4.9231043e-03,
						7.4080727e-03, -1.1844908e-02, 2.9069011e-03, 4.0808925e-03, -5.6662667e-03,
						6.2518441e-03, 8.5856044e-04, 4.8197974e-03, 3.5093622e-03, -5.4735183e-03,
					}),
				},
				params.Param{
					Weight: mat.NewDense(5, 7, []float64{
						-0.01282741, -0.00087706, -0.01778904, 0.01260526, -0.00340842, -0.00538257, 0.01233118,
						-0.0040539, -0.00133679, -0.00904377, -0.0113096, 0.00994975, 0.01763666, -0.00049345,
						-0.00247867, 0.00963731, -0.00886456, 0.00564355, -0.00867464, 0.01505886, 0.00194833,
						0.0126702, -0.00274673, 0.01662341, 0.00127717, 0.01869018, -0.00081351, -0.00134613,
						-0.0003356, 0.00964021, 0.00460173, 0.01488965, -0.01594927, 0.00531581, 0.02366197,
					}),
				},
			},
			wantGrads: []params.Grad{
				params.Grad{
					Weight: mat.NewDense(7, 5, []float64{
						-0.00021922, 0.00025485, -0.00131402, 0.00151378, -0.00061075,
						0.00259923, 0.00153943, 0.00176951, -0.0017144, 0.00022873,
						-0.00268571, 0.00217196, -0.00196254, 0.00235711, -0.00209673,
						0.00259923, 0.00153943, 0.00176951, -0.0017144, 0.00022873,
						-0.00246649, 0.00191711, -0.00064852, 0.00084333, -0.00148598,
						0., 0., 0., 0., 0.,
						0., 0., 0., 0., 0.,
					}),
				},
				params.Grad{
					Weight: mat.NewDense(5, 7, []float64{
						-7.23263197e-04, -1.88345180e-05, 1.94964953e-03, 9.62016851e-04, -7.23232408e-04, -7.23225987e-04, -7.23110269e-04,
						-1.64165643e-05, 7.89892226e-04, -1.26347251e-04, -5.97862361e-04, -1.63935278e-05, -1.64230318e-05, -1.64494897e-05,
						6.64884889e-04, -2.12321939e-03, 3.97048203e-04, -9.33246055e-04, 6.64817737e-04, 6.64879019e-04, 6.64835598e-04,
						8.99976459e-04, -1.76340025e-03, 3.60499897e-05, -1.87235652e-03, 8.99913672e-04, 8.99956947e-04, 8.99859699e-04,
						2.50602983e-04, -2.42614640e-03, 3.88623330e-04, 1.03511475e-03, 2.50530315e-04, 2.50615260e-04, 2.50659754e-04,
					}),
				},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			gotParams, gotGrads := trainer.RmDuplicate(tt.params, tt.grads)

			if len(gotParams) != len(tt.wantParams) {
				t.Fatalf("unexpected lendth: want: %v, got: %v", len(tt.wantParams), len(gotParams))
			}
			if len(gotGrads) != len(tt.wantGrads) {
				t.Fatalf("unexpected lendth: want: %v, got: %v", len(tt.wantGrads), len(gotGrads))
			}

			for i, w := range tt.wantParams {
				if !mat.EqualApprox(gotParams[i].Weight, w.Weight, 1e-2) {
					t.Errorf("x:\nwant = %d\ngot = %d", w.Weight, gotParams[i].Weight)
				}
			}
			for i, g := range tt.wantGrads {
				if !mat.EqualApprox(gotGrads[i].Weight, g.Weight, 1e-2) {
					t.Errorf("x:\nwant = %d\ngot = %d", g.Weight, gotGrads[i].Weight)
				}
			}
		})
	}
}

func Test3DFit(t *testing.T) {
	windowSize := 1
	hiddenSize := 5
	batchSize := 3
	maxEpoch := 1

	text := "You say goodbye and I say hello."
	corpus, w2id, _ := word.PreProcess(text)

	vocabSize := len(w2id)
	contexts, target := word.CreateContextsAndTarget(corpus, windowSize)

	te := word.ConvertOneHot(target, vocabSize)
	co := word.ConvertOneHot(contexts, vocabSize)

	model := models.InitSimpleCBOW(vocabSize, hiddenSize)
	optimizer := optimizers.InitAdam(0.001, 0.9, 0.999)
	trainer := trainer.InitTrainer(model, optimizer)

	// checks no panic ...
	trainer.Fit3D(co, matutil.At3D(te, 0), maxEpoch, batchSize)
	_ = trainer.GetWordDist()
}
