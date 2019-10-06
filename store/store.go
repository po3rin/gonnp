// Package store lets you to store trained data.
package store

import (
	"encoding/gob"
	"log"
	"os"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

func init() {
	// regist mat.Matrix data structure.
	gob.Register(mat.Matrix(mat.DenseCopyOf(mat.NewDense(1, 1, nil))))
}

// CBOW is store of CBOW output.
type CBOW struct {
	W2ID     map[string]float64
	ID2W     map[float64]string
	WordVecs mat.Matrix
}

// NewCBOWEncoder new CBOW output for encoding.
func NewCBOWEncoder(w2id map[string]float64, id2w map[float64]string, wordVecs mat.Matrix) *CBOW {
	return &CBOW{
		W2ID:     w2id,
		ID2W:     id2w,
		WordVecs: wordVecs,
	}
}

// Encode CBOW output to file.
func (c *CBOW) Encode(fileName string) error {
	f, err := os.Create(fileName)
	if err != nil {
		log.Fatal(err)
	}
	err = gob.NewEncoder(f).Encode(&c)
	if err != nil {
		return errors.Wrap(err, "gonnp: failed to encode CBOWOutput struct")
	}
	return nil
}

// Decode CBOW output file to struct.
func (c *CBOW) Decode(fileName string) error {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}

	err = gob.NewDecoder(f).Decode(c)
	if err != nil {
		return errors.Wrap(err, "gonnp: failed to dencode file to CBOWOutput struct")
	}
	return nil
}
