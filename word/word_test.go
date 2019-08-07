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
			wantTarget:   mat.NewDense(7, 1, []float64{1, 2, 3, 4, 1, 5, 6}),
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
