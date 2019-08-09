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
		if !containsString(wordToID, word) {
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
func CreateContextsAndTarget(corpus Corpus) (contexts, target mat.Matrix) {
	var windowSize = 1
	ts := corpus[windowSize : len(corpus)-1]
	cs := make([]float64, 0, len(corpus)*2-2)

	for i := windowSize; i < len(corpus)-windowSize; i++ {
		for j := -windowSize; j < windowSize+1; j++ {
			if j == 0 {
				continue
			}
			cs = append(cs, corpus[i+j])
		}
	}
	return mat.NewDense(len(cs)/2, 2, cs), mat.NewVecDense(len(ts), ts)
}

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
