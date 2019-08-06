package word

import "strings"

// Corpus type is include id num only.
type Corpus []int

// Word2ID for changing word to id.
type Word2ID map[string]int

// ID2Word for changing id to word.
type ID2Word map[int]string

// PreProcess create corpus, wordToID, idToWprd.
func PreProcess(text string) (Corpus, Word2ID, ID2Word) {
	text = strings.ToLower(text)
	text = strings.ReplaceAll(text, ".", " .")
	words := strings.Split(text, " ")

	wordToID := make(Word2ID, len(words))
	idToWord := make(ID2Word, len(words))

	for _, word := range words {
		if !containsString(wordToID, word) {
			newID := len(wordToID)
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
