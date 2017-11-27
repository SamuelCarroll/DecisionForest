package main

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"time"

	"github.com/SamuelCarroll/DataTypes"
	"github.com/SamuelCarroll/DecisionTree"
	"github.com/SamuelCarroll/readFile"

	"gonum.org/v1/gonum/mat"
)

//CLASSES is the number of classes we have for a particular dataset
const CLASSES = 3

func main() {
	var decTree DecisionTree.Tree
	var decForest []DecisionTree.Tree
	setVal := 100000000000.0 //big value to ignore a already used split feature value
	stopCond := 0.84         //point were we stop training if we have this percent of a single class
	rand.Seed(time.Now().UTC().UnixNano())

	allData := readFile.Read("/home/ritadev/Documents/Thesis_Work/Decision-Tree/wine.data")

	//call bagging, get back a slice of training data and a slice of testing data
	trainSets, testSets := bagging(allData)

	// uncomment the following line to test the reading of a Decision Tree
	// testRead(allData)

	start := time.Now()
	for _, trainData := range trainSets {
		decTree = decTree.Train(trainData, setVal, stopCond, CLASSES)
		decForest = append(decForest, decTree)
	}
	elapsed := time.Since(start)
	fmt.Println("It took ", elapsed, " to train the forest")

	// totalElems, totalMisclassified := 0, 0
	// for i, testData := range testSets {
	misclassified := 0
	// fmt.Printf("Test Set: %d\n", i)
	// fmt.Printf("+-----------+----------+\n")
	// fmt.Printf("| Predicted |  Actual  |\n")
	// fmt.Printf("+-----------+----------+\n")

	start = time.Now()
	for _, elem := range testSets /*testData*/ {
		var guesses []int
		for _, tree := range decForest {
			estimatedClass := tree.GetClass(*elem)

			guesses = append(guesses, estimatedClass)
		}

		prediction := getMajority(guesses)
		if prediction != elem.Class {
			misclassified++
		}
		// fmt.Printf("|     %d     |     %d    |\n", prediction, elem.Class)
	}
	elapsed = time.Since(start)
	fmt.Println("It took ", elapsed, " to test the forest")

	// fmt.Printf("+-----------+----------+\n")

	fmt.Printf("%d out of %d wrongly classified\n", misclassified, len(testSets) /*len(testData)*/)
	// fmt.Printf("Misclassified: %f%%\n", float64(misclassified)/float64(len(testData))*100.0)
	fmt.Printf("Misclassified: %f%%\n", float64(misclassified)/float64(len(testSets)))
	// totalElems += len(testSets) //len(testData)
	// totalMisclassified += misclassified
	// }
	// fmt.Println("Final Forest Results:")
	// fmt.Printf("%d out of %d wrongly classified\n", totalMisclassified, totalElems)
	// fmt.Printf("Misclassified: %f%%\n", float64(totalMisclassified)/float64(totalElems)*100.0)

	var fastDensities [][]float64
	start = time.Now()
	for _, trainSet := range trainSets {
		_, den := getRFDiss(decForest, trainSet)
		fastDensities = append(fastDensities, den)
	}
	elapsed = time.Since(start)
	fmt.Println("It took ", elapsed, " to generate the diss matrix")

	// Uncomment the following lines to write the DF to files
	// for i, tree := range decForest {
	// 	tree.WriteTree("tree" + strconv.Itoa(i) + ".txt")
	// }
}

//TODO create some test conditions to see if we have this programmed correctly
func getRFDiss(decForest []DecisionTree.Tree, trainData []*dataTypes.Data) (*mat.Dense, []float64) {
	numTrees := len(decForest)
	dataLen := len(trainData)
	treeResults := make([]*DecisionTree.Tree, numTrees*dataLen)
	rfSlice := make([]float64, dataLen*dataLen)

	for i, tree := range decForest {
		for j, datum := range trainData {
			treeResults[(i*dataLen)+j] = tree.GetTerminalNode(*datum)
		}
	}

	for i := 0; i < dataLen; i++ {
		elem1 := treeResults[i]
		for j, elem2 := range treeResults {
			jMod := j % dataLen
			if elem1 == elem2 {
				rfSlice[i*dataLen+jMod] += 1.0
			}
		}
	}

	for i, val := range rfSlice {
		rfSlice[i] = math.Sqrt(1.0 - (val / float64(numTrees)))
	}

	rfMat := mat.NewDense(dataLen, dataLen, rfSlice)

	return rfMat, rfSlice
}

//TODO generate a []*dataTypes.Data of synthetic data labeled 1 append to
//observed and return
func genSynthetic(observed []*dataTypes.Data) []*dataTypes.Data {
	for _, observation := range observed {
		observation.Class = 0
	}

	return observed
}

func testRead(dataSet []*dataTypes.Data) {
	var decForest []DecisionTree.Tree
	misclassified := 0

	for i := 0; i < 5; i++ {
		var tempTree DecisionTree.Tree
		err := tempTree.ReadTree("tree" + strconv.Itoa(i) + ".txt")
		if err != nil {
			fmt.Println(err)
			return
		}
		decForest = append(decForest, tempTree)
	}

	fmt.Printf("+-----------+----------+\n")
	fmt.Printf("| Predicted |  Actual  |\n")
	fmt.Printf("+-----------+----------+\n")

	for _, elem := range dataSet {
		var guesses []int
		for _, tree := range decForest {
			estimatedClass := tree.GetClass(*elem)

			guesses = append(guesses, estimatedClass)
		}

		prediction := getMajority(guesses)
		if prediction != elem.Class {
			misclassified++
		}
		fmt.Printf("|     %d     |     %d    |\n", prediction, elem.Class)
	}
	fmt.Printf("+-----------+----------+\n")

	fmt.Printf("%d out of %d wrongly classified\n", misclassified, len(dataSet))
	fmt.Printf("Misclassified: %f%%\n", float64(misclassified)/float64(len(dataSet))*100.0)
}

func bagging(allData []*dataTypes.Data) ([][]*dataTypes.Data, []*dataTypes.Data) {
	dataLen := len(allData)

	//30 Training error went between ~6.2% and ~39.2% Average around 20.699% error
	//25 Training error went between ~7.5% and ~16.5% Average around 11.896% error
	//20 Training error went between ~7.7% and ~16.0% Average around 10.939% error
	//15 Training error was between ~5% and ~14.5% Average around 12.39% error
	//10 Training error was between ~6.9% and ~12.0% Average around 9.330% error
	//5 Training error was between ~1.2% and ~10.8% Average around 8.2
	kclassifiers := 500
	var trainSets [][]*dataTypes.Data
	var testSets []*dataTypes.Data

	//Generate a number of sets to train different trees on that data
	for i := 0; i < kclassifiers; i++ {
		//randomly select an index from 0-dataLen add that element to the end of a tempset
		//at the end of that
		var newTestSet []*dataTypes.Data
		var newTrainSet []*dataTypes.Data
		var usedIndices []int

		for len(newTrainSet) < dataLen {
			randIndex := rand.Intn(dataLen)
			usedIndices = append(usedIndices, randIndex)
			newTrainSet = append(newTrainSet, allData[randIndex])
		}

		newTestSet = getNewTestSet(allData, usedIndices, dataLen)
		trainSets = append(trainSets, newTrainSet)
		testSets = append(testSets, newTestSet...)
	}

	return trainSets, testSets
}

func getNewTestSet(allData []*dataTypes.Data, usedIndices []int, dataLen int) []*dataTypes.Data {
	var newSet []*dataTypes.Data
	for j := 0; j < dataLen; j++ {
		indexUsed := false
		for _, usedIndex := range usedIndices {
			if usedIndex == j {
				indexUsed = true
				break
			}
		}
		if !indexUsed {
			newSet = append(newSet, allData[j])
		}
	}
	return newSet
}

//TODO change this from a simple majority vote to a Baysian network analysis of the
//accuracy of the tree for the training data
func getMajority(guesses []int) int {
	var classGuesses []int

	for _, guess := range guesses {
		if (guess) > len(classGuesses) {
			for (guess) > len(classGuesses) {
				classGuesses = append(classGuesses, 0)
			}
		}
		classGuesses[guess-1]++
	}

	maxIndex := 1

	for i, contender := range classGuesses {
		if contender > classGuesses[maxIndex-1] {
			maxIndex = i + 1
		}
	}

	return maxIndex
}
