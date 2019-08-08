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
		Iter       float64
		m          []mat.Matrix
		v          []mat.Matrix
		params     []entity.Param
		grads      []entity.Grad
		wantParams []entity.Param
	}{
		{
			name:  "real value",
			lr:    0.001,
			beta1: 0.9,
			beta2: 0.999,
			Iter:  1999,
			m: []mat.Matrix{
				mat.NewDense(7, 5, []float64{
					0.00128465, -0.00848898, -0.01045923, -0.00108253, 0.00142057,
					-0.00369151, 0.00326042, -0.00270434, 0.00397773, -0.0036535,
					0.00346632, 0.00978515, 0.01233115, -0.00346579, 0.00338299,
					-0.00188448, -0.02162165, 0.00246958, 0.00201432, -0.00184111,
					0.00372328, 0.00812329, 0.01028116, -0.00368306, 0.00366662,
					0.00154161, -0.01015084, -0.01250921, -0.00129981, 0.0017042,
					-0.00180703, 0.02488208, -0.00517392, 0.0019634, -0.00181239,
				}),
				mat.NewDense(5, 7, []float64{
					-0.00055318, 0.0047359, -0.00222076, 0.00697386, -0.00245548, -0.00593052, -0.00054983,
					0.00048814, -0.00834097, -0.01418194, 0.01035775, -0.01452606, 0.02572849, 0.00047458,
					0.00119593, -0.01254619, 0.00263549, 0.01646944, 0.00245858, -0.011413, 0.00119975,
					0.00057293, -0.00444617, 0.00236244, -0.00732289, 0.00260133, 0.00566265, 0.00056972,
					-0.00042667, 0.00492711, -0.0022839, 0.0066257, -0.00252432, -0.00589452, -0.00042341,
				}),
			},
			v: []mat.Matrix{
				mat.NewDense(7, 5, []float64{
					1.9614918e-04, 8.0861995e-04, 9.0201828e-04, 1.4872888e-04, 2.2831510e-04,
					2.6381426e-04, 2.7802214e-03, 3.3331013e-04, 3.2635487e-04, 2.3475963e-04,
					4.0934762e-04, 1.6718069e-03, 2.6539846e-03, 3.5236738e-04, 4.4303949e-04,
					1.1351515e-04, 2.1540979e-03, 8.1511913e-05, 1.3020988e-04, 9.5652125e-05,
					4.1726153e-04, 1.6212706e-03, 2.5727598e-03, 3.5902637e-04, 4.5169698e-04,
					1.9728570e-04, 8.1770046e-04, 9.1371808e-04, 1.4953168e-04, 2.2967842e-04,
					6.7964662e-05, 3.1058039e-03, 2.0302266e-04, 9.2710528e-05, 6.5178618e-05,
				}),
				mat.NewDense(5, 7, []float64{
					8.29980272e-05, 2.99606565e-03, 1.25837438e-02, 2.96134385e-03, 1.26792975e-02, 2.35447031e-03, 8.20337082e-05,
					4.93213920e-05, 8.22601025e-04, 1.13085341e-02, 3.29224218e-04, 1.12228161e-02, 2.11184472e-03, 4.89494923e-05,
					8.77638959e-05, 8.24994349e-04, 1.00305285e-02, 1.08394516e-03, 1.00922249e-02, 2.69475277e-03, 8.85798363e-05,
					7.76581219e-05, 2.76215980e-03, 1.28638428e-02, 2.89932708e-03, 1.29585629e-02, 2.33496819e-03, 7.69242106e-05,
					8.01798597e-05, 2.90550571e-03, 1.30312685e-02, 2.81218160e-03, 1.31320059e-02, 2.37426115e-03, 7.92823048e-05,
				}),
			},
			params: []entity.Param{
				entity.Param{
					Weight: mat.NewDense(7, 5, []float64{
						-0.9310736, 1.3598018, 1.5747849, 0.87879294, -0.92997706,
						1.160113, 0.2969906, 1.1542624, -1.1721987, 1.1808438,
						-1.1574188, -0.21978574, -0.66300726, 1.1773701, -1.1343554,
						0.96708906, 1.8049119, 0.49000862, -0.97432154, 0.99166995,
						-1.158001, -0.20263301, -0.65530926, 1.1668633, -1.109128,
						-0.929702, 1.3517597, 1.5568544, 0.91460544, -0.9206439,
						1.0512303, -1.6471609, 1.3513477, -1.0230746, 1.0605339,
					}),
				},
				entity.Param{
					Weight: mat.NewDense(5, 7, []float64{
						0.25506052, -0.970362, 0.64586055, -0.9080978, 0.6517141, 0.7960724, 0.25100636,
						-1.1342931, 1.268895, 0.86125946, -1.5132158, 0.8563066, -2.1796303, -1.1416876,
						-1.2790875, 1.4306083, 0.29309803, -1.9559118, 0.29289708, 0.7038156, -1.2761831,
						-0.0846772, 0.99007875, -0.6220933, 0.98623526, -0.61727995, -0.76922673, -0.09519083,
						0.30299267, -0.98184407, 0.65683204, -0.8790631, 0.6605833, 0.8106478, 0.2960545,
					}),
				},
			},
			grads: []entity.Grad{
				entity.Grad{
					Weight: mat.NewDense(7, 5, []float64{
						0.00275694, -0.01826525, -0.0225004, -0.002323, 0.003049,
						-0.00422985, -0.04826664, 0.00551606, 0.00452217, -0.00413453,
						0.00275694, -0.01826525, -0.0225004, -0.002323, 0.003049,
						-0.00422985, -0.04826664, 0.00551606, 0.00452217, -0.00413453,
						0., 0., 0., 0., 0.,
						0., 0., 0., 0., 0.,
						0., 0., 0., 0., 0.,
					}),
				},
				entity.Grad{
					Weight: mat.NewDense(5, 7, []float64{
						0.00223061, 0.02254175, -0.02280207, -0.01217709, -0.02287244, 0.03086405, 0.0022152,
						0.00312585, -0.00161311, -0.02096735, 0.00668951, -0.02104925, 0.03070584, 0.0031085,
						0.00245103, -0.00141079, -0.01639254, 0.00534975, -0.01645669, 0.02402179, 0.00243745,
						-0.00226562, -0.022364, 0.02298412, 0.01198791, 0.02305532, -0.03114768, -0.00225004,
						0.00229787, 0.02251008, -0.02325428, -0.01203531, -0.02332641, 0.03152596, 0.00228209,
					}),
				},
			},
			wantParams: []entity.Param{
				entity.Param{
					Weight: mat.NewDense(7, 5, []float64{
						-0.93116874, 1.3601115, 1.5751461, 0.878885, -0.9300746,
						1.1603276, 0.29702398, 1.1543584, -1.1724063, 1.1810685,
						-1.157575, -0.21994455, -0.66316706, 1.1775361, -1.1345035,
						0.96727407, 1.8053985, 0.4897228, -0.9745062, 0.9918669,
						-1.1581535, -0.20280194, -0.655479, 1.167026, -1.1092725,
						-0.9297939, 1.352057, 1.5572009, 0.9146944, -0.9207381,
						1.0514139, -1.6475347, 1.3516518, -1.0232453, 1.0607219,
					}),
				},
				entity.Param{
					Weight: mat.NewDense(5, 7, []float64{
						0.2550886, -0.97054696, 0.645896, -0.9081843, 0.6517512, 0.7961156, 0.25103444,
						-1.1343927, 1.2691438, 0.86138946, -1.513728, 0.8564399, -2.1801612, -1.1417857,
						-1.2792188, 1.4309787, 0.29309124, -1.9563458, 0.29289183, 0.7039566, -1.276314,
						-0.08470771, 0.9901892, -0.6221296, 0.9863284, -0.6173179, -0.7692649, -0.09522136,
						0.3030087, -0.98195946, 0.65686774, -0.87914664, 0.6606207, 0.81068885, 0.2960705,
					}),
				},
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			optimizer := optimizers.InitAdam(tt.lr, tt.beta1, tt.beta2)

			optimizer.Iter = tt.Iter
			optimizer.M = tt.m
			optimizer.V = tt.v

			got := optimizer.Update(tt.params, tt.grads)
			if len(got) != len(tt.wantParams) {
				t.Fatalf("unexpected lendth: want: %v, got: %v", len(tt.wantParams), len(got))
			}

			for i, w := range tt.wantParams {
				if !mat.EqualApprox(got[i].Weight, w.Weight, 1e-2) {
					t.Errorf("x:\nwant = %d\ngot = %d", w.Weight, got[i].Weight)
				}
			}
		})
	}
}
