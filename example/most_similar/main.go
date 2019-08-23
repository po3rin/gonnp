package main

import (
	"os"

	"github.com/po3rin/gonnp/store"
	"github.com/po3rin/gonnp/word"
)

// print most similar word using CBOW model.
func main() {
	cbow := &store.CBOW{}
	cbow.Decode("testdata/cbow.gob")

	word.WriteMostSimilar(os.Stdout, "you", cbow.W2ID, cbow.ID2W, cbow.WordVecs)
}
