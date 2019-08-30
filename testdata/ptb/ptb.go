package ptb

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/po3rin/gonnp/word"
	"golang.org/x/sync/errgroup"
)

func loadVocab(dir string) (word.Word2ID, word.ID2Word, error) {
	path := filepath.Join(dir, "ptb.train.txt")
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, nil, err
	}
	text := string(bytes)

	text = strings.ReplaceAll(text, "\n", "")
	text = strings.ReplaceAll(text, "<eos>", "")
	words := strings.Split(text, " ")

	wordToID := make(word.Word2ID, len(words))
	idToWord := make(word.ID2Word, len(words))

	for _, word := range words {
		_, ok := wordToID[word]
		if !ok {
			newID := float64(len(wordToID))
			wordToID[word] = newID
			idToWord[newID] = word
		}
	}

	return wordToID, idToWord, nil
}

// LoadData loads PTB dataset. dataType: 'train' or 'test' or 'valid'.
func LoadData(dir, dataType string) (word.Corpus, word.Word2ID, word.ID2Word) {
	var w2id word.Word2ID
	var id2w word.ID2Word
	var words []string

	eg := errgroup.Group{}
	eg.Go(func() error {
		var err error
		w2id, id2w, err = loadVocab(dir)
		if err != nil {
			return err
		}
		return nil
	})

	eg.Go(func() error {
		path := filepath.Join(dir, "ptb."+dataType+".txt")
		bytes, err := ioutil.ReadFile(path)
		if err != nil {
			return err
		}
		text := string(bytes)
		text = strings.ReplaceAll(text, "\n", "")
		text = strings.ReplaceAll(text, "<eos>", "")
		words = strings.Split(text, " ")
		words = words[:len(words)-1] // rm last word ""
		return nil
	})

	if err := eg.Wait(); err != nil {
		log.Fatal(err)
	}

	corpus := make(word.Corpus, 0, len(words))
	for _, word := range words {
		corpus = append(corpus, w2id[word])
	}

	return corpus, w2id, id2w
}
