// +build !e2e

package word_test

import (
	"reflect"
	"testing"

	"github.com/po3rin/gonlp/word"
	"gonum.org/v1/gonum/mat"
)

func TestPreProcess(t *testing.T) {
	tests := []struct {
		name       string
		text       string
		wantCorpus word.Corpus
		wantW2ID   word.Word2ID
		wantID2W   word.ID2Word
	}{
		{
			name:       "simple",
			text:       "You say goodbye and I say hello.",
			wantCorpus: []float64{0, 1, 2, 3, 4, 1, 5, 6},
			wantW2ID: map[string]float64{
				".": 6, "and": 3, "goodbye": 2, "hello": 5, "i": 4, "say": 1, "you": 0,
			},
			wantID2W: map[float64]string{
				0: "you", 1: "say", 2: "goodbye", 3: "and", 4: "i", 5: "hello", 6: ".",
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			corpus, w2id, id2w := word.PreProcess(tt.text)

			if !reflect.DeepEqual(tt.wantCorpus, corpus) {
				t.Errorf("want = %v, got = %v\n", tt.wantCorpus, corpus)
			}
			if !reflect.DeepEqual(tt.wantW2ID, w2id) {
				t.Errorf("want = %v, got = %v\n", tt.wantW2ID, w2id)
			}
			if !reflect.DeepEqual(tt.wantID2W, id2w) {
				t.Errorf("want = %v, got = %v\n", tt.wantID2W, id2w)
			}
		})
	}
}

func TestCreateContextsAndTarget(t *testing.T) {
	tests := []struct {
		name         string
		corpus       word.Corpus
		wantContexts mat.Matrix
		wantTarget   mat.Matrix
	}{
		{
			name:         "simple",
			corpus:       []float64{0, 1, 2, 3, 4, 1, 5, 6},
			wantContexts: mat.NewDense(6, 2, []float64{0, 2, 1, 3, 2, 4, 3, 1, 4, 5, 1, 6}),
			wantTarget:   mat.NewDense(6, 1, []float64{1, 2, 3, 4, 1, 5}),
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			contexts, target := word.CreateContextsAndTarget(tt.corpus)

			if !mat.EqualApprox(contexts, tt.wantContexts, 1e-7) {
				t.Errorf("x:\nwant = %d\ngot = %d", tt.wantContexts, contexts)
			}
			if !mat.EqualApprox(target, tt.wantTarget, 1e-7) {
				t.Errorf("x:\nwant = %d\ngot = %d", tt.wantTarget, target)
			}
		})
	}
}

func TestConvertOneHot(t *testing.T) {
	tests := []struct {
		name      string
		corpus    mat.Matrix
		vacabSize int
		want      []mat.Matrix
	}{
		{
			name:      "simple",
			corpus:    mat.NewDense(6, 1, []float64{1, 2, 3, 4, 1, 5}),
			vacabSize: 7,
			want: []mat.Matrix{
				mat.NewDense(1, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 1, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 1, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 0, 1, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(1, 7, []float64{
					0, 0, 0, 0, 0, 1, 0,
				}),
			},
		},
		{
			name: "3 dimention",
			corpus: mat.NewDense(6, 2, []float64{
				0, 2,
				1, 3,
				2, 4,
				3, 1,
				4, 5,
				1, 6,
			}),
			vacabSize: 7,
			want: []mat.Matrix{
				mat.NewDense(2, 7, []float64{
					1, 0, 0, 0, 0, 0, 0,
					0, 0, 1, 0, 0, 0, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
					0, 0, 0, 1, 0, 0, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 0, 1, 0, 0, 0, 0,
					0, 0, 0, 0, 1, 0, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 0, 0, 1, 0, 0, 0,
					0, 1, 0, 0, 0, 0, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 0, 0, 0, 1, 0, 0,
					0, 0, 0, 0, 0, 1, 0,
				}),
				mat.NewDense(2, 7, []float64{
					0, 1, 0, 0, 0, 0, 0,
					0, 0, 0, 0, 0, 0, 1,
				}),
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			got := word.ConvertOneHot(tt.corpus, tt.vacabSize)

			if len(got) != len(tt.want) {
				t.Fatalf("unexpected lendth: want: %v, got: %v", len(tt.want), len(got))
			}

			for i, w := range tt.want {
				if !mat.EqualApprox(got[i], w, 1e-7) {
					t.Errorf("x:\nwant = %d\ngot = %d", w, got[i])
				}
			}
		})
	}
}

func TestGetWord2VecFromDist(t *testing.T) {
	tests := []struct {
		name string
		id2w word.ID2Word
		dist mat.Matrix
		want map[string]mat.Vector
	}{
		{
			name: "simple",
			id2w: map[float64]string{
				0: "you", 1: "say", 2: "goodbye", 3: "and", 4: "i", 5: "hello", 6: ".",
			},
			dist: mat.NewDense(7, 5, []float64{
				-0.9790874903673588, -1.025641372026828, 1.0095565367722954, -1.0504750829640517, -1.6481770684712491,
				0.7724434672379474, 1.386847010669654, -1.4066035838270725, -0.009356605377383642, -0.3176750750342113,
				-1.0337446638191161, -1.0489240882343083, 1.0164216809676263, -0.8763330333430552, 0.7132255901200747,
				-1.5107822855008428, 1.2447710975076722, -1.2541383173900162, -1.4276743837936225, -1.2792412897834655,
				-1.0013253401759428, -1.047856555127371, 1.0282795434962002, -0.8965252994562523, 0.7104409303964825,
				-0.9901973257219888, -1.0136141693951204, 1.010225251104932, -1.0455978503391465, -1.6396751010503963,
				1.2085340320721714, 1.1096341070890634, -1.1331510149004795, 1.2617809111710436, 1.3524260611358265,
			}),
			want: map[string]mat.Vector{
				"you": mat.NewVecDense(5, []float64{
					-0.9790874903673588, -1.025641372026828, 1.0095565367722954, -1.0504750829640517, -1.6481770684712491,
				}),
				"say": mat.NewVecDense(5, []float64{
					0.7724434672379474, 1.386847010669654, -1.4066035838270725, -0.009356605377383642, -0.3176750750342113,
				}),
				"goodbye": mat.NewVecDense(5, []float64{
					-1.0337446638191161, -1.0489240882343083, 1.0164216809676263, -0.8763330333430552, 0.7132255901200747,
				}),
				"and": mat.NewVecDense(5, []float64{
					-1.5107822855008428, 1.2447710975076722, -1.2541383173900162, -1.4276743837936225, -1.2792412897834655,
				}),
				"i": mat.NewVecDense(5, []float64{
					-1.0013253401759428, -1.047856555127371, 1.0282795434962002, -0.8965252994562523, 0.7104409303964825,
				}),
				"hello": mat.NewVecDense(5, []float64{
					-0.9901973257219888, -1.0136141693951204, 1.010225251104932, -1.0455978503391465, -1.6396751010503963,
				}),
				".": mat.NewVecDense(5, []float64{
					1.2085340320721714, 1.1096341070890634, -1.1331510149004795, 1.2617809111710436, 1.3524260611358265,
				}),
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			got := word.GetWord2VecFromDist(tt.dist, tt.id2w)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("x:\nwant = %+v\ngot = %+v", tt.want, got)
			}
		})
	}
}
