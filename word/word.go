// Package word is functions of text processing.
package word

import (
	"strings"

	"gonum.org/v1/gonum/mat"
)

// Corpus type is include id num only.
type Corpus []float64

// Word2ID for changing word to id.
type Word2ID map[string]float64

// ID2Word for changing id to word.
type ID2Word map[float64]string

// PreProcess create corpus, wordToID, idToWprd.
func PreProcess(text string) (Corpus, Word2ID, ID2Word) {
	text = strings.ToLower(text)
	text = strings.ReplaceAll(text, ".", " .")
	words := strings.Split(text, " ")

	wordToID := make(Word2ID, len(words))
	idToWord := make(ID2Word, len(words))

	for _, word := range words {
		_, ok := wordToID[word]
		if !ok {
			newID := float64(len(wordToID))
			wordToID[word] = newID
			idToWord[newID] = word
		}
	}

	corpus := make(Corpus, 0, len(words))
	for _, word := range words {
		corpus = append(corpus, wordToID[word])
	}

	return corpus, wordToID, idToWord
}

// CreateContextsAndTarget creates contexts and target from text corpus.
func CreateContextsAndTarget(corpus Corpus, windowSize int) (contexts, target mat.Matrix) {
	ts := corpus[windowSize : len(corpus)-windowSize]
	cs := make([]float64, 0, len(corpus)*2-2)

	for i := windowSize; i < len(corpus)-windowSize; i++ {
		for j := -windowSize; j < windowSize+1; j++ {
			if j == 0 {
				continue
			}
			cs = append(cs, corpus[i+j])
		}
	}

	cr := mat.NewDense(len(cs)/(windowSize*2), windowSize*2, cs)
	tr := mat.NewVecDense(len(ts), ts)
	return cr, tr
}

// ConvertOneHot converts corpus to one-hot-matrix.
func ConvertOneHot(corpus mat.Matrix, vocabSize int) []mat.Matrix {
	r, c := corpus.Dims()
	ts := make([]mat.Matrix, 0, vocabSize-1)

	for i := 0; i < r; i++ {
		t := mat.NewDense(c, vocabSize, nil)
		for j := 0; j < c; j++ {
			v := corpus.At(i, j)
			t.Set(j, int(v), 1)
		}
		ts = append(ts, t)
	}
	return ts
}

// GetWord2VecFromDist convert Distributed representation to word-vec map.
func GetWord2VecFromDist(dist mat.Matrix, id2w ID2Word) map[string]mat.Vector {
	r, _ := dist.Dims()
	w2v := make(map[string]mat.Vector, r)
	d, ok := dist.(*mat.Dense)
	if !ok {
		panic("gonnp: failed to transpose matrix to dense")
	}
	for i := 0; i < r; i++ {
		w2v[id2w[float64(i)]] = d.RowView(i)
	}
	return w2v
}
