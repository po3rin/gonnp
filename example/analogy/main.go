package main

import (
	"log"

	"github.com/po3rin/gonnp/store"
	"github.com/po3rin/gonnp/word"
)

// anology word.
func main() {
	cbow := &store.CBOW{}
	cbow.Decode("testdata/cbow.gob")

	_, err := word.Analogy("man", "king", "women", cbow.W2ID, cbow.ID2W, cbow.WordVecs)
	if err != nil {
		log.Fatal(err)
	}

	_, err = word.Analogy("take", "took", "go", cbow.W2ID, cbow.ID2W, cbow.WordVecs)
	if err != nil {
		log.Fatal(err)
	}

	_, err = word.Analogy("car", "cars", "child", cbow.W2ID, cbow.ID2W, cbow.WordVecs)
	if err != nil {
		log.Fatal(err)
	}

	_, err = word.Analogy("good", "better", "bad", cbow.W2ID, cbow.ID2W, cbow.WordVecs)
	if err != nil {
		log.Fatal(err)
	}
}
