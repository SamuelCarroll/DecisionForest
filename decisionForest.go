package DecisionForest

import (
	"fmt"
	"math"
	"math/rand"
	"strconv"
	"time"

	"github.com/SamuelCarroll/DataTypes"
	"github.com/SamuelCarroll/DecisionTree"

	"gonum.org/v1/gonum/mat"
)

//GenForest builds a decision tree of a specified size with a given number of classes and returns the forest and the data used to test the tree
func GenForest(allData []*dataTypes.Data, numClasses, numTrees int, printRes, writeTrees, readTrees bool) ([]DecisionTree.Tree, []*dataTypes.Data) {
	var decTree DecisionTree.Tree
	var decForest []DecisionTree.Tree
	setVal := 100000000000.0 //big value to ignore a already used split feature value
	stopCond := 0.84         //point were we stop training if we have this percent of a single class
	rand.Seed(time.Now().UTC().UnixNano())

	//call bagging, get back a slice of training data and a slice of testing data
	trainSets, testSets := bagging(allData, numTrees)

	if readTrees == true {
		testRead(allData)
	}

	start := time.Now()
	for _, trainData := range trainSets {
		decTree = decTree.Train(trainData, setVal, stopCond, numClasses)
		decForest = append(decForest, decTree)
	}
	elapsed := time.Since(start)
	fmt.Println("It took ", elapsed, " to train the forest")

	misclassified := 0
	if printRes == true {
		fmt.Printf("+-----------+----------+\n")
		fmt.Printf("| Predicted |  Actual  |\n")
		fmt.Printf("+-----------+----------+\n")
		start = time.Now()
	}

	for _, elem := range testSets {
		var guesses []int
		for _, tree := range decForest {
			estimatedClass := tree.GetClass(*elem)

			guesses = append(guesses, estimatedClass)
		}

		prediction := getMajority(guesses)
		if prediction != elem.Class {
			misclassified++
		}
		if printRes {
			fmt.Printf("|     %d     |     %d    |\n", prediction, elem.Class)
		}
	}
	if printRes {
		elapsed = time.Since(start)
		fmt.Printf("+-----------+----------+\n")

		fmt.Printf("%d out of %d wrongly classified\n", misclassified, len(testSets) /*len(testData)*/)
		fmt.Printf("Misclassified: %f%%\n", (float64(misclassified) / float64(len(testSets)) * 100.0))

		fmt.Println("It took ", elapsed, " to test the forest")
	}

	if writeTrees {
		for i, tree := range decForest {
			tree.WriteTree("tree" + strconv.Itoa(i) + ".txt")
		}
	}

	return decForest, testSets
}

//GenMatrix trains a Decision Forest and then find a Disimilarity Matrix
func GenMatrix(allData []*dataTypes.Data, numClasses, numTrees int) (*mat.Dense, []float64) {

	decForest, testSets := GenForest(allData, numClasses, numTrees, false, false, false)

	start := time.Now()
	// _, den := getRFDiss(decForest, testSets)
	rfMat, rfSlice := getRFDiss(decForest, testSets)
	elapsed := time.Since(start)
	fmt.Println("It took ", elapsed, " to generate the diss matrix")

	return rfMat, rfSlice
}

//TODO create some test conditions to see if we have this programmed correctly
func getRFDiss(decForest []DecisionTree.Tree, trainData []*dataTypes.Data) (*mat.Dense, []float64) {
	numTrees := len(decForest)
	dataLen := len(trainData)
	treeResults := make([]*DecisionTree.Tree, numTrees*dataLen)
	rfSlice := make([]float64, dataLen*dataLen)

	//start by getting the addresses of the nodes each result ends up in
	for i, tree := range decForest {
		for j, datum := range trainData {
			treeResults[(i*dataLen)+j] = tree.GetTerminalNode(*datum)
		}
	}

	//Run through the previous table to calculate the similarity between each element in each tree
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

func bagging(allData []*dataTypes.Data, numTrees int) ([][]*dataTypes.Data, []*dataTypes.Data) {
	dataLen := len(allData)
	dataUsed := make([]bool, len(allData))

	var trainSets [][]*dataTypes.Data
	var testSets []*dataTypes.Data

	//Generate a number of sets to train different trees on that data
	for i := 0; i < numTrees; i++ {
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

		newTestSet, dataUsed = getNewTestSet(allData, usedIndices, dataLen, dataUsed)
		trainSets = append(trainSets, newTrainSet)
		testSets = append(testSets, newTestSet...)
	}

	return trainSets, testSets
}

func getNewTestSet(allData []*dataTypes.Data, usedIndices []int, dataLen int, dataUsed []bool) ([]*dataTypes.Data, []bool) {
	var newSet []*dataTypes.Data
	for j := 0; j < dataLen; j++ {
		indexUsed := false
		for _, usedIndex := range usedIndices {
			if usedIndex == j {
				indexUsed = true
				dataUsed[usedIndex] = true
				break
			}
		}
		if !indexUsed && dataUsed[j] == false {
			newSet = append(newSet, allData[j])
		}
	}
	return newSet, dataUsed
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
