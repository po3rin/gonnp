package word_test

import (
	"bytes"
	"testing"

	"github.com/po3rin/gonnp/store"
	"github.com/po3rin/gonnp/word"
	"gonum.org/v1/gonum/mat"
)

func TestCosSimilarity(t *testing.T) {
	tests := []struct {
		name string
		vec1 mat.Vector
		vec2 mat.Vector
		want float64
	}{
		{
			name: "same",
			vec1: mat.NewVecDense(5, []float64{0, 0, 1, 0, 0}),
			vec2: mat.NewVecDense(5, []float64{0, 0, 1, 0, 0}),
			want: 0.9999999800000005,
		},
		{
			name: "same",
			vec1: mat.NewVecDense(5, []float64{0, 1, 1, 0, 0}),
			vec2: mat.NewVecDense(5, []float64{0, 0, 1, 1, 0}),
			want: 0.49999999292893216,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := word.CosSimilarity(tt.vec1, tt.vec2)
			if tt.want != got {
				t.Errorf("want: %v, got: %v\n", tt.want, got)
			}
		})
	}
}

func TestWriteMostSimilar(t *testing.T) {

	tests := []struct {
		query string
		want  string
	}{
		{
			query: "year",
			want: `month: 0.8566038553796044
week: 0.7842923540411619
summer: 0.7676705352324136
spring: 0.7637303121661754
decade: 0.7004195797345385
minute: 0.5673860636841538
`,
		},
		{
			query: "you",
			want: `we: 0.744096372149493
i: 0.7307638803440424
your: 0.6390780189216694
they: 0.6117267722023164
someone: 0.5924593205450905
us: 0.5862648616313337
`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.query, func(t *testing.T) {
			cbow := &store.CBOW{}
			cbow.Decode("./../testdata/cbow.gob")

			stdout := new(bytes.Buffer)
			word.WriteMostSimilar(stdout, tt.query, cbow.W2ID, cbow.ID2W, cbow.WordVecs)
			if tt.want != stdout.String() {
				t.Errorf("want:\n%v\ngot :\n%v\n", tt.want, stdout.String())
			}
		})
	}
}

// func TestAnalogy(t *testing.T) {
// 	tests := []struct {
// 		name string
// 		a    string
// 		b    string
// 		c    string
// 		want string
// 		w2id word.Word2ID
// 		id2w word.ID2Word
// 	}{
// 		{
// 			name: "simple",
// 			a:    "king",
// 			b:    "man",
// 			c:    "queen",
// 			want: "woman",
// 		},
// 	}

// 	for _, tt := range tests {
// 		t.Run(tt.name, func(t *testing.T) {
// 			cbow := &store.CBOW{}
// 			cbow.Decode("./../testdata/cbow.gob")

// 			got, err := word.Analogy(tt.a, tt.b, tt.c, cbow.W2ID, cbow.ID2W, cbow.WordVecs)
// 			if err != nil {
// 				t.Fatalf("unexpected error: %+v\n", err)
// 			}
// 			if tt.want != got {
// 				t.Errorf("want: %v, got: %v\n", tt.want, got)
// 			}
// 		})
// 	}
// }
