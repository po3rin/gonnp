// +build !e2e

package word_test

import (
	"reflect"
	"testing"

	"github.com/po3rin/gonlp/word"
)

func TestPreProcess(t *testing.T) {
	tests := []struct {
		name       string
		text       string
		wantCorpus word.Corpus
		wantW2ID   word.Word2ID
		wantID2W   word.ID2Word
	}{
		{
			name:       "simple",
			text:       "You say goodbye and I say hello.",
			wantCorpus: []int{0, 1, 2, 3, 4, 1, 5, 6},
			wantW2ID: map[string]int{
				".": 6, "and": 3, "goodbye": 2, "hello": 5, "i": 4, "say": 1, "you": 0,
			},
			wantID2W: map[int]string{
				0: "you", 1: "say", 2: "goodbye", 3: "and", 4: "i", 5: "hello", 6: ".",
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			corpus, w2id, id2w := word.PreProcess(tt.text)

			if !reflect.DeepEqual(tt.wantCorpus, corpus) {
				t.Errorf("want = %v, got = %v\n", tt.wantCorpus, corpus)
			}
			if !reflect.DeepEqual(tt.wantW2ID, w2id) {
				t.Errorf("want = %v, got = %v\n", tt.wantW2ID, w2id)
			}
			if !reflect.DeepEqual(tt.wantID2W, id2w) {
				t.Errorf("want = %v, got = %v\n", tt.wantID2W, id2w)
			}
		})
	}
}
