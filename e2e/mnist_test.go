package e2e_test

import (
	"testing"

	"github.com/po3rin/gomnist"
	"github.com/po3rin/gonlp/nn"
	"github.com/po3rin/gonlp/optimizers"
	"github.com/po3rin/gonlp/trainer"
)

func TestMNITS(t *testing.T) {
	model := nn.NewTwoLayerNet(2, 10, 3)
	optimizer := optimizers.InitSDG(1)
	trainer := trainer.InitTrainer(model, optimizer, trainer.EvalInterval(10))

	l := gomnist.NewLoader("./testdata")
	mnist, err := l.Load()
	if err != nil {
		t.Fatalf("unexpected error: %v\n", err)
	}

	_ = trainer
	_ = mnist

	// trainer.Fit(mnist.TestData, mnist.TestLabels, 100, 30)
}
