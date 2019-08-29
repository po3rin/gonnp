package word

import (
	"fmt"
	"io"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

type similar struct {
	word  string
	score float64
}

type similarList []similar

func (s similarList) Len() int           { return len(s) }
func (s similarList) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s similarList) Less(i, j int) bool { return s[i].score > s[j].score }

// TODO: refactoring...
func cosSimilarity(x, y mat.Vector) float64 {
	xd, ok := x.(*mat.VecDense)
	if !ok {
		panic("gonnp: failed to transpose matrix to dense")
	}
	yd, ok := y.(*mat.VecDense)
	if !ok {
		panic("gonnp: failed to transpose matrix to dense")
	}

	r, _ := x.Dims()

	var sumx, sumy float64
	for i := 0; i < r; i++ {
		sumx += math.Pow(xd.AtVec(i), 2)
		sumy += math.Pow(yd.AtVec(i), 2)
	}

	sqx := math.Sqrt(sumx)
	sqy := math.Sqrt(sumy)

	sqx += 1e-8
	sqy += 1e-8

	xd.ScaleVec(1/sqx, xd)
	yd.ScaleVec(1/sqy, yd)

	xd.MulElemVec(xd, yd)
	return mat.Sum(xd)
}

// WriteMostSimilar returns most similar word.
func WriteMostSimilar(w io.Writer, query string, w2id Word2ID, id2w ID2Word, wordMatrix mat.Matrix) {
	_, ok := w2id[query]
	if !ok {
		fmt.Printf("%s is not found\n", query)
	}

	d, ok := wordMatrix.(*mat.Dense)
	if !ok {
		panic("gonnp: failed to transpose matrix to dense")
	}
	fmt.Printf("[query] %s\n", query)
	queryID := w2id[query]
	queryVec := d.RowView(int(queryID))

	vocabSize := len(id2w)

	list := make(similarList, vocabSize)
	for i := 0; i < vocabSize; i++ {
		w := id2w[float64(i)]
		if w == query {
			continue
		}
		list[i] = similar{
			word:  id2w[float64(i)],
			score: cosSimilarity(d.RowView(i), queryVec),
		}
	}

	sort.Sort(list)

	for i, v := range list {
		result := fmt.Sprintf("%s: %v\n", v.word, v.score)
		w.Write([]byte(result))
		if i > 4 {
			break
		}
	}
}

// type simMaps []simMap
// type simMap struct {
// 	id int
// 	m  float64
// }

// func (s simMaps) Len() int           { return len(s) }
// func (s simMaps) Less(i, j int) bool { return s[i].m > s[j].m }
// func (s simMaps) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// // Analogy analogies word vector. a + b = c + result .
// func Analogy(a, b, c string, w2id Word2ID, id2w ID2Word, wordMatrix mat.Matrix) (string, error) {
// 	aID, ok := w2id[a]
// 	if !ok {
// 		return "", fmt.Errorf("gonnp: %v is not found in words", a)
// 	}
// 	bID, ok := w2id[b]
// 	if !ok {
// 		return "", fmt.Errorf("gonnp: %v is not found in words", b)
// 	}
// 	cID, ok := w2id[c]
// 	if !ok {
// 		return "", fmt.Errorf("gonnp: %v is not found in words", c)
// 	}

// 	wm, ok := wordMatrix.(*mat.Dense)
// 	if !ok {
// 		panic("gonnp: failed to transpose matrix to dense")
// 	}

// 	av := wm.RowView(int(aID))
// 	bv := wm.RowView(int(bID))
// 	cv := wm.RowView(int(cID))

// 	avd, ok := av.(*mat.VecDense)
// 	if !ok {
// 		panic("gonnp: failed to transpose matrix to dense")
// 	}
// 	bvd, ok := bv.(*mat.VecDense)
// 	if !ok {
// 		panic("gonnp: failed to transpose matrix to dense")
// 	}
// 	cvd, ok := cv.(*mat.VecDense)
// 	if !ok {
// 		panic("gonnp: failed to transpose matrix to dense")
// 	}

// 	avd.AddVec(avd, cvd)
// 	avd.SubVec(bvd, avd)

// 	queryVec := avd
// 	queryVec = matutils.NormoalizeVec(queryVec)

// 	mr, _ := wordMatrix.Dims()
// 	similarity := mat.NewDense(mr, 1, nil)
// 	similarity.Product(wordMatrix, queryVec)

// 	sm := make(simMaps, mr)
// 	for i := 0; i < mr; i++ {
// 		sm[i] = simMap{
// 			id: i,
// 			m:  similarity.At(i, 0),
// 		}
// 	}

// 	sort.Sort(sm)

// 	for i := 0; i < len(sm); i++ {
// 		fmt.Println("--------------")
// 		if i > 100 {
// 			break
// 		}
// 		result, _ := id2w[float64(sm[i].id)]
// 		fmt.Println(sm[i].id)
// 		fmt.Println(sm[i].m)
// 		fmt.Println(result)
// 	}

// 	result, ok := id2w[float64(sm[0].id)]
// 	if !ok {
// 		return "", fmt.Errorf("gonnp: %v is not found in id2w", sm[0].id)
// 	}
// 	return result, nil
// }
