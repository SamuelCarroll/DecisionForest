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

func boolFloatToStr(boolFloat float64) string {
	if boolFloat == 0.0 {
		return "False"
	}
	return "True"
}

func getString(datum dataTypes.Data, prediction int) []string {
	var datumStr []string

	datumStr = append(datumStr, strconv.Itoa(prediction))
	datumStr = append(datumStr, datum.UID)

	for _, feature := range datum.FeatureSlice {
		var appStr string
		switch feature.(type) {
		case float64:
			datum := DecisionTree.GetFloatReflectVal(feature)
			appStr = strconv.FormatFloat(datum, 'f', 24, 64)
		case bool:
			datumBool := DecisionTree.GetBoolReflectVal(feature)
			appStr = boolFloatToStr(datumBool)
		case string:
			appStr = DecisionTree.GetStrReflectVal(feature)
		}
		datumStr = append(datumStr, appStr)
	}

	return datumStr
}

func labelData(forest []DecisionTree.Tree, newData []*dataTypes.Data) []*dataTypes.Data {
	for i, elem := range newData {
		var guesses []int
		for _, tree := range forest {
			estimatedClass := tree.GetClass(*elem, i)

			guesses = append(guesses, estimatedClass)
		}

		newData[i].Class = getMajority(guesses)
	}

	return newData
}

//SemiSupervisedLearning will implement a SSL method
func SemiSupervisedLearning(labeledData, unlabeledData []*dataTypes.Data, numClasses, numTrees, generation int, printRes, writeTrees, readTrees bool, outBase string) ([]DecisionTree.Tree, []*dataTypes.Data) {
	var supervisedForest []DecisionTree.Tree
	//If this is the first generation we should create the inital forest
	//If it's not the first generation we should read in the forest
	if generation == 0 {
		supervisedForest, _ = GenForest(labeledData, numClasses, numTrees, printRes, writeTrees, readTrees, outBase)
	} else {
		supervisedForest = testRead(labeledData, false, outBase, generation*numTrees)
	}

	newData := labelData(supervisedForest, unlabeledData)

	for _, newDatum := range newData {
		labeledData = append(labeledData, newDatum)
	}

	finalForest, finalData := GenForest(labeledData, numClasses, numTrees, printRes, writeTrees, readTrees, outBase)

	return finalForest, finalData
}

//GenForest builds a decision tree of a specified size with a given number of classes and returns the forest and the data used to test the tree
func GenForest(allData []*dataTypes.Data, numClasses, numTrees int, printRes, writeTrees, readTrees bool, outBase string) ([]DecisionTree.Tree, []*dataTypes.Data) {
	var decTree DecisionTree.Tree
	var decForest []DecisionTree.Tree
	setVal := 100000000000.0 //big value to ignore a already used split feature value
	stopCond := 0.85         //point were we stop training if we have this percent of a single class
	rand.Seed(time.Now().UTC().UnixNano())

	//if we want to specify to read previously made trees just test that forest
	//with all that data
	if readTrees == true {
		testRead(allData, printRes, outBase, numTrees)
		return nil, nil
	}

	//TODO see if we get speed boost by modifying bagging to return a single
	//training set, put it in loop
	//call bagging, get back a slice of training data and a slice of testing data
	trainSets, testSets := bagging(allData, numTrees)

	//get the start time of training/building the forest so we know how long it takes
	start := time.Now()
	//For each bagging training set we generated make a tree
	for _, trainData := range trainSets {
		decTree = decTree.Train(trainData, setVal, stopCond, numClasses)
		decForest = append(decForest, decTree)
	}
	elapsed := time.Since(start)
	fmt.Println("It took ", elapsed, " to train the forest")

	//Start testing on the OOB data
	misclassified := 0
	if printRes == true {
		//fmt.Printf("+-----------+----------+-------------------------+\n")
		//fmt.Printf("| Predicted |  Actual  |           UID   \t |\n")
		//fmt.Printf("+-----------+----------+-------------------------+\n")
		start = time.Now()
	}

	//For every element in the OOB set run it through every tree in the forest
	//To classify that given element
	var allPredictions []int
	for i, elem := range testSets {
		var guesses []int
		for _, tree := range decForest {
			estimatedClass := tree.GetClass(*elem, i)

			guesses = append(guesses, estimatedClass)
		}

		prediction := getMajority(guesses)
		if prediction != elem.Class {
			misclassified++
		}
		//if printRes {
		//fmt.Printf("|     %d     |     %d    |   %s\t |", prediction, elem.Class, elem.UID)
		//if prediction == 1 && elem.Class == 2 {
		//	fmt.Printf("\t oops")
		//}
		//fmt.Printf("\n")
		//}
		if writeTrees {
			allPredictions = append(allPredictions, prediction)
		}
	}
	//Print the end of the data if we specify print
	if printRes {
		elapsed = time.Since(start)
		//fmt.Printf("+-----------+----------+-------------------------+\n")

		fmt.Printf("%d out of %d wrongly classified\n", misclassified, len(testSets) /*len(testData)*/)
		fmt.Printf("Misclassified: %f%%\n", (float64(misclassified) / float64(len(testSets)) * 100.0))

		fmt.Println("It took ", elapsed, " to test the forest")
	}

	//If we want to write the trees to the current directory
	if writeTrees {
		for i, tree := range decForest {
			tree.WriteTree(outBase + strconv.Itoa(i) + ".txt")
		}
	}

	return decForest, testSets
}

//GenMatrix trains a Decision Forest and then find a Disimilarity Matrix
func GenMatrix(allData []*dataTypes.Data, numClasses, numTrees int) (*mat.Dense, []float64, int) {

	//Before we can generate the dissimilarity matrix we need to generate the forest
	decForest, _ := GenForest(allData, numClasses, numTrees, false, false, false, "")

	//Get the start time of generating the matrix
	start := time.Now()
	// _, den := getRFDiss(decForest, testSets)
	rfMat, rfSlice := getRFDiss(decForest, allData)
	elapsed := time.Since(start)
	fmt.Println("It took ", elapsed, " to generate the diss matrix")

	//Return the Matrix, the slice representation of the matrix and how many data we have
	return rfMat, rfSlice, len(allData)
}

//TODO create some test conditions to see if we have this programmed correctly
//getRFDiss will generate the Random Forest dissimilarity matrix and return it
func getRFDiss(decForest []DecisionTree.Tree, data []*dataTypes.Data) (*mat.Dense, []float64) {
	//Get the number of trees in the forest and the count of data
	//also generate a slice to hold which node each element will lay in for a given tree
	//also create a slice to hold the similarity matrix
	numTrees := len(decForest)
	dataLen := len(data)
	treeResults := make([]*DecisionTree.Tree, numTrees*dataLen)
	rfSlice := make([]float64, dataLen*dataLen)

	//start by getting the addresses of the nodes each result ends up in
	for i, tree := range decForest {
		for j, datum := range data {
			treeResults[(i*dataLen)+j] = tree.GetTerminalNode(*datum, j)
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

	//Find the dissimilarity by subtracting the similarity from 1
	for i, val := range rfSlice {
		rfSlice[i] = math.Sqrt(1.0 - (val / float64(numTrees)))
	}

	//Generate the matrix from the slic
	rfMat := mat.NewDense(dataLen, dataLen, rfSlice)

	return rfMat, rfSlice
}

//This will test a forest that is stored in a series of files
func testRead(dataSet []*dataTypes.Data, printRes bool, outBase string, numTrees int) []DecisionTree.Tree {
	var decForest []DecisionTree.Tree
	misclassified := 0
	//Positive is anomalous, negative is normal
	truePositive := 0
	trueNegative := 0
	falsePositive := 0
	falseNegative := 0

	//Read in all trees and add each tree to the forest
	for i := 0; i < numTrees; i++ {
		var tempTree DecisionTree.Tree
		err := tempTree.ReadTree(outBase + strconv.Itoa(i) + ".txt")
		if err != nil {
			fmt.Println(err)
			return nil
		}
		decForest = append(decForest, tempTree)
	}

	fmt.Println("Forest Read")

	start := time.Now()

	//If this is set to true we assume we want to build this all over again
	if printRes == true {
		// fmt.Printf("+-----------+----------+-------------------------+\n")
		// fmt.Printf("| Predicted |  Actual  |           UID   \t |\n")
		// fmt.Printf("+-----------+----------+-------------------------+\n")

		//For each datum run it through the forest, writing the results
		tenPercent := len(dataSet) / 10
		datLen := len(dataSet)
		for i, elem := range dataSet {
			var guesses []int

			if i%tenPercent == 0 {
				fmt.Println(100*i/datLen, "% done")
			}

			for _, tree := range decForest {
				estimatedClass := tree.GetClass(*elem, i)

				guesses = append(guesses, estimatedClass)
			}

			prediction := getMajority(guesses)
			if prediction == 1 && elem.Class == 1 {
				trueNegative++
			} else if prediction == 2 && elem.Class == 2 {
				truePositive++
			} else if prediction == 1 && elem.Class == 2 {
				falseNegative++
			} else if prediction == 2 && elem.Class == 1 {
				falsePositive++
			}

			if prediction != elem.Class {
				misclassified++
			}
			//fmt.Printf("|     %d     |     %d    |   %s\t |\n", prediction, elem.Class, elem.UID)
		}
		//fmt.Printf("+-----------+----------+-------------------------+\n")
		elapsed := time.Since(start)

		fmt.Printf("%d out of %d wrongly classified\n", misclassified, len(dataSet))
		fmt.Printf("Misclassified: %f%%\n", (float64(misclassified)/float64(len(dataSet)))*100.0)
		//Positive is anomalous, negative is normal
		fmt.Printf("\tAnom Correctly Labeled: %d\n\tNorm Correctly Labeled: %d\n", truePositive, trueNegative)
		fmt.Printf("\tAnom incorrectly Labeled: %d\n\tNorm incorrectly Labeled: %d\n", falseNegative, falsePositive)
		fmt.Println("It took", elapsed, "to test", len(dataSet), "elements")
	}

	return decForest
}

//bagging will randomly generate a series of different data to train the forest
//and to test the forest when it is trained
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

		//While we don't have all the training sets make generate a random index and
		//add it to used indicies list add the corresponding datum to a training set
		//NOTE: we want to do this with replacement, each datum can be selected
		//multiple times
		for len(newTrainSet) < dataLen {
			randIndex := rand.Intn(dataLen)
			usedIndices = append(usedIndices, randIndex)
			newTrainSet = append(newTrainSet, allData[randIndex])
		}

		//generate the test dataset, add trainset to training sets, add to testset
		newTestSet, dataUsed = getNewTestSet(allData, usedIndices, dataLen, dataUsed)
		trainSets = append(trainSets, newTrainSet)
		testSets = append(testSets, newTestSet...)
	}

	return trainSets, testSets
}

//getNewTestSet will use all the data read from a file, the used indicies to
//generate the test set for use after we generate the forest
func getNewTestSet(allData []*dataTypes.Data, usedIndices []int, dataLen int, dataUsed []bool) ([]*dataTypes.Data, []bool) {
	//initialize the newSet to be empty, we will fill it up and return it
	var newSet []*dataTypes.Data
	//for each datum in the data we will check if it's index is used, appending it
	//to our test set if it hasn't been used
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
//getMajority will get the results from every tree and guess which class the
//datum belongs to by a majority vote
func getMajority(guesses []int) int {
	var classGuesses []int

	//tally each guess into the proper class guess
	for _, guess := range guesses {
		if (guess) > len(classGuesses) {
			for (guess) > len(classGuesses) {
				classGuesses = append(classGuesses, 0)
			}
		}
		classGuesses[guess-1]++
	}

	maxIndex := 1

	//find the largest value, compensate for off by one error and return
	for i, contender := range classGuesses {
		if contender > classGuesses[maxIndex-1] {
			maxIndex = i + 1
		}
	}

	return maxIndex
}
