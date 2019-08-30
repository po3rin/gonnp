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
			want: `month: 0.8577425321209501
week: 0.7834646860168318
spring: 0.7763444531620151
summer: 0.7677653678415696
decade: 0.7095253509774642
minute: 0.577743504663146
`,
		},
		{
			query: "you",
			want: `we: 0.7330695462803781
i: 0.7094418928150142
your: 0.6334623096987391
they: 0.5993890779126023
someone: 0.5880683533754492
anybody: 0.5758252412919126
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

func TestAnalogy(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		a       string
		b       string
		c       string
		wordMat mat.Matrix
		want    string
	}{
		{
			name: "fake data",
			text: "you say goodbye i say",
			a:    "say",
			b:    "you",
			c:    "say",
			wordMat: mat.NewDense(5, 5, []float64{
				0.5942552, 0.73378074, 0.76838416, 0.7570169, -0.76511973,
				-0.79442763, -0.77520937, -0.80227834, -0.7986194, 0.78109694,
				0.8141478, 0.8137531, 0.79203695, 0.8052735, -0.8325244,
				-0.79765093, -0.7953921, -0.7788316, -0.7819185, 0.77854705,
				0.80453324, 0.73260766, 0.70120907, 0.7057159, -0.7123809,
			}),
			want: "goodbye",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, w2id, id2w := word.PreProcess(tt.text)

			got, err := word.Analogy(tt.a, tt.b, tt.c, w2id, id2w, tt.wordMat)
			if err != nil {
				t.Fatalf("unexpected error: %+v\n", err)
			}
			if tt.want != got {
				t.Errorf("want: %v, got: %v\n", tt.want, got)
			}
		})
	}
}
