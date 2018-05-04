package de.proai.nn;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

public class CsvTest {

	public static void main(String[] args) throws FileNotFoundException {
	
		double[][][] data = loadData(new File("dataSet_aktuell_julian.csv"));
		
		NeuralNetwork nn = new NeuralNetwork(new int[] {4, 7, 1}, new Random());
		
		for(int epoch=0; epoch<10000; epoch++) {
			
			if(epoch%1000==0)
				System.out.println(epoch);
			
			nn.train(data, 1);
		}
		
		for(int entry=0; entry<data.length; entry++) {
			double[] input = data[entry][0];
			double[] output = nn.calculate(input);
			double[] target = data[entry][1];
			
			for(int i=0; i<output.length; i++) {
				output[i] = output[i] < .5 ? 0 : 1;
			}
			
//			double error = 0;
//			for(int i=0; i<output.length; i++) {
//				error += Math.pow((output[i] - target[i]), 2);
//			}
//			error /= output.length;
			
			boolean correct = true; 
			
			for(int i=0; i<output.length; i++) {
				correct &= output[i] == target[i];
			}
					
			System.out.println(Arrays.toString(input) + ": \t" + Arrays.toString(output) + "; \t" + Arrays.toString(target) + "  correct: " + correct);
		}
	}
	
	public static double[][][] loadData(File file) throws FileNotFoundException {
		
		Scanner scanner = new Scanner(file);
		
		ArrayList<ArrayList<Integer>> rows = new ArrayList<>();
		
		while(scanner.hasNextLine()) {
			String line = scanner.nextLine();
			String[] parts = line.split(",");
			
			ArrayList<Integer> row = new ArrayList<>();
			for(String part : parts) {
				if(!part.equals("")) {
					Integer val = Integer.parseInt(part);
					row.add(val);
				}
			}
			
			rows.add(row);
		}
		
		scanner.close();
		
		double[][][] data = new double[rows.size()][][];
		int i = 0;
		for(ArrayList<Integer> row : rows) {
			double[][] dataRow = new double[2][];
			dataRow[0] = new double[4];
			dataRow[1] = new double[1];
			
			for(int j=0; j<4; j++) {
				double val = row.get(j);
				
				if(j == 0)
					val = 0;
				else if(j == 1)
					val /= 1100;
				else if(j == 2)
					val /= 20;
				else if(j == 3)
					val /= 110;
				
				dataRow[0][j] = val;
			}
			
			dataRow[1][0] = row.get(10);
			
			data[i++] = dataRow;
		}
	
		return data;
	}
}
