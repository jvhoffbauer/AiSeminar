package de.proai.nn;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class IrisTest {

	public static void main(String[] args) {
		
		double[][][] data = loadData();
		int epochs = 50000;
		int[] layerDefs = new int[] {4, 8, 3};
		double learningRate = 0.01;

		List<double[][]> randomizing = new ArrayList<double[][]>(Arrays.asList(data));
		Collections.shuffle(randomizing);
		data = randomizing.toArray(data);
		
		doKFold(data, 0,    30, epochs, layerDefs, learningRate);
		doKFold(data, 30,   60, epochs, layerDefs, learningRate);
		doKFold(data, 60,   90, epochs, layerDefs, learningRate);
		doKFold(data, 90,  120, epochs, layerDefs, learningRate);
		doKFold(data, 120, 150, epochs, layerDefs, learningRate);
	}
	
	/**
	 * Do k-fold training step
	 * @param data
	 * @param evalBegin
	 * @param evalEnd
	 * @param epochs
	 * @param layerDefs
	 * @param learningRate
	 */
	private static void doKFold(double[][][] data, int evalBegin, int evalEnd, int epochs, int[] layerDefs, double learningRate) {
		
		System.out.println("KFold: " + evalBegin + " - " + evalEnd);
		
		ArrayList<double[][]> trainingDataList = new ArrayList<>();
		ArrayList<double[][]> evalDataList = new ArrayList<>();
		
		// Extract training and evaluation data
		for(int entry=0; entry<data.length; entry++) {
			
			if(entry >= evalBegin && entry < evalEnd) 
				evalDataList.add(data[entry]);
			else
				trainingDataList.add(data[entry]);
		}
		
		// Train
		double[][][] trainingData = trainingDataList.toArray(new double[trainingDataList.size()][][]);
		NeuralNetwork nn = new NeuralNetwork(layerDefs);
		for(int epoch=0; epoch<epochs; epoch++) {
			nn.train(trainingData, learningRate);
		}
		
		// Evaluate
		int numcorrect = 0;
		double sumError = 0;
		for(int entry=0; entry<evalDataList.size(); entry++) {
			double[] input = evalDataList.get(entry)[0];
			double[] output = nn.calculate(input);
			double[] target = evalDataList.get(entry)[1];
			
			double error = 0;
			for(int i=0; i<output.length; i++) {
				error += Math.pow((output[i] - target[i]), 2);
			}
			error /= output.length;
			
			boolean isCorrect = true;
			for(int i=0; i<output.length; i++) {
				if( (output[i] < .5 ? 0 : 1) != target[i] )
					isCorrect = false;
			}
			if(isCorrect)
				numcorrect++;
			
			sumError += error;
		}
		System.out.println("  Accuracy: " + (numcorrect / (double) evalDataList.size()));
		System.out.println("  Error:    " + (sumError / evalDataList.size()));
	}
	
	/**
	 * Load dataset from file
	 * @return
	 */
	private static double[][][] loadData() {
		
		ArrayList<double[]> inputList = new ArrayList<>();
		ArrayList<double[]> outputList = new ArrayList<>();
		
		Scanner in = null;
		try {
			in = new Scanner(new File("iris.data"));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		
		while(in.hasNextLine()) {
			String line = in.nextLine();
			String[] parts = line.split(",");

			double d1 = Double.parseDouble(parts[0]);
			double d2 = Double.parseDouble(parts[1]);
			double d3 = Double.parseDouble(parts[2]);
			double d4 = Double.parseDouble(parts[3]);
			inputList.add(new double[] {d1, d2, d3, d4});

			if(parts[4].equals("Iris-setosa")) 
				outputList.add(new double[] {0.0, 0.0, 1.0});
			else if(parts[4].equals("Iris-versicolor")) 
				outputList.add(new double[] {0.0, 1.0, 0.0});
			else if(parts[4].equals("Iris-virginica")) 
				outputList.add(new double[] {1.0, 0.0, 0.0});
		}
		
		double[][][] data = new double[outputList.size()][][];
		for(int d=0; d<data.length; d++) {
			data[d] = new double[2][];
			data[d][0] = inputList.get(d);
			data[d][1] = outputList.get(d);
		}
		
		return data;
	}
}
