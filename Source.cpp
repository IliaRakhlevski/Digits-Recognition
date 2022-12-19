#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <windows.h>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <stdlib.h>
#include <math.h>       /* atan */
using namespace cv;
using namespace std;


//////////////  GLOBAL DEFINITIONS  ////////////////

#define PI 3.14159265

// HISTOGRAM
#define N_BUCKETS  18		// number of buckets in histogram
#define CELL_SIZE	8		// Each cell is 8x8 pixels
#define BLOCK_SIZE  4		// Each block is 4x4 cells


Mat image, gray;				// loaded image: original and grayscale
Mat G_x;						// edges - filtered image: x axis
Mat G_y;						// edges - filtered image: y axis
double bucket_vals[N_BUCKETS];	// buckets containing magnitudes
double block_bucket_vals[N_BUCKETS * BLOCK_SIZE * BLOCK_SIZE];	// block of buckets


// NETWORK
#define INPUT_SIZE  (N_BUCKETS * BLOCK_SIZE * BLOCK_SIZE)	// input layer size
#define HIDDEN_SIZE INPUT_SIZE								// hidden layer size
#define OUTPUT_SIZE 10										// digits 0 - 9


#define TETHA  (double)0.01

// found number by recognition and real (target) number
unsigned int found_number;		// recognized digit index in output[]
unsigned int real_number = 0;	// real digit

// input values - histogram data
double input[INPUT_SIZE];

// hidden layer
double hidden[HIDDEN_SIZE];

// output layer
double output[OUTPUT_SIZE];

// target values
double target[OUTPUT_SIZE];

// weights between input layer to hidden one
double weights_in_hid[INPUT_SIZE][HIDDEN_SIZE];

// weights between hidden layer to output one
double weights_hid_out[HIDDEN_SIZE][OUTPUT_SIZE];

// values for weights fixing
double delta_weights_hid_out[HIDDEN_SIZE][OUTPUT_SIZE];
double delta_weights_in_hid[INPUT_SIZE][HIDDEN_SIZE];

// values for weight adjusting
double delta[OUTPUT_SIZE];

string fonts[] = {"Arial", "Tnr", "Col", "David", "Cent", "Bahn", "Imp"}; // fonts names

#define NUM_IMG_FOR_DIGIT_TRAIN	(sizeof(fonts) / sizeof(fonts[0])) // number of fonts in fonts[]



////////////////  GLOBAL FUNCTIONS ////////////////


/////////////// HISTOGRAM functions  /////////////

// assign new value to relevant bucket: m - magnitude, d - direction (grad)
void assign_bucket_vals(double m, double d)
{
	if (d >= 20 * N_BUCKETS)
		d = d - (20 * N_BUCKETS);

	// find two neighbors buckets
	int right_bin = int(d / 20.0);
	int left_bin = (int(d / 20.0) + 1);

	// split magnitude value between two neighbors buckets
	double right_val = m * (left_bin * 20.0 - d) / 20.0;
	double left_val = m * (d - right_bin * 20.0) / 20.0;

	if (left_bin == N_BUCKETS)
		left_bin = 0;

	// assign magnitude values values to buckets
	bucket_vals[right_bin] += right_val;
	bucket_vals[left_bin] += left_val;
}


// create histogram for one cell starting in [loc_x, loc_y]
void get_magnitude_hist_cell(int loc_x, int loc_y)
{
	double magnitudes[CELL_SIZE][CELL_SIZE] = { 0 };
	double directions[CELL_SIZE][CELL_SIZE] = { 0 };

	// calculate magnitudes and directions
	for (int x = loc_x, i = 0; x < loc_x + CELL_SIZE; x++, i++)
	{
		for (int y = loc_y, j = 0; y < loc_y + CELL_SIZE; y++, j++)
		{
			magnitudes[i][j] = sqrt(G_x.at<int16_t>(x, y) * G_x.at<int16_t>(x,y) + G_y.at<int16_t>(x, y) * G_y.at<int16_t>(x, y));
			directions[i][j] = abs(atan2(abs(G_y.at<int16_t>(x, y)), abs(G_x.at<int16_t>(x, y))) * 180.0 / PI);
		}
	}

	// adjust directions to 0 - 360 grad scale
	for (int x = loc_x, i = 0; x < loc_x + CELL_SIZE; x++, i++)
	{
		for (int y = loc_y, j = 0; y < loc_y + CELL_SIZE; y++, j++)
		{
			if (G_x.at<int16_t>(x, y) < 0 && G_y.at<int16_t>(x, y) >= 0)
				directions[i][j] = 180.0 - directions[i][j];
			else if (G_x.at<int16_t>(x, y) < 0 && G_y.at<int16_t>(x, y) < 0)
				directions[i][j] += 180.0;
			else if (G_x.at<int16_t>(x, y) >= 0 && G_y.at<int16_t>(x, y) < 0)
				directions[i][j] = 360 - directions[i][j];
		}
	}

	memset(bucket_vals, 0, sizeof(double) * N_BUCKETS);

	// assign values to buckets
	for (int i = 0; i < CELL_SIZE; i++)
		for (int j = 0; j < CELL_SIZE; j++)
			assign_bucket_vals(magnitudes[i][j], directions[i][j]);

}


// create histograms for all cells in the block
void get_magnitude_hist_block(int loc_x, int loc_y)
{
	int count = 0;

	// get buckets with relevant values for each cell in block
	for (int i = 0; i < BLOCK_SIZE; i++)
	{
		for (int j = 0; j < BLOCK_SIZE; j++)
		{
			get_magnitude_hist_cell(j * CELL_SIZE, i * CELL_SIZE);
			memcpy(&block_bucket_vals[count * N_BUCKETS], bucket_vals, N_BUCKETS * sizeof(double));
			count++;
		}
	}

	// frobenius normalization of buckets values
	double norm = 0.0;
	for (int i = 0; i < (N_BUCKETS * BLOCK_SIZE * BLOCK_SIZE); i++)
		norm += (block_bucket_vals[i] * block_bucket_vals[i]);
	norm = sqrt(norm);
	for (int i = 0; i < (N_BUCKETS * BLOCK_SIZE * BLOCK_SIZE); i++)
		block_bucket_vals[i] /= norm;
}


// find edges
void FilterEdges()
{
	// create filters for getting edges and perform filtering
	Mat kern_x = (Mat_<char>(1, 3) << -1, 0, 1);
	//cout << "X filter:" << endl << endl << kern_x << endl << endl;
	filter2D(gray, G_x, CV_16S, kern_x);
	//cout << "X edges:" << endl << endl << G_x << endl << endl;

	Mat kern_y = (Mat_<char>(3, 1) << 1, 0, -1);
	//cout << "Y filter:" << endl << endl << kern_y << endl << endl;
	filter2D(gray, G_y, CV_16S, kern_y);
	//cout << "Y edges:" << endl << endl << G_y << endl << endl;
}


// load image from file
void LoadImage(String imageName)
{
	image = imread(imageName); // Read the file

	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		exit(0);
	}

	//cout << image << endl;

	//namedWindow("Display window", WINDOW_AUTOSIZE); // Create a window for display.
	//imshow("Display window", image);                // Show our image inside it.

	// get gray scale image
	cvtColor(image, gray, COLOR_BGR2GRAY);

	//imshow("Display window", gray);


	// resize the gray scale image to size: 32 x 32 
	if (image.rows != 32 || image.cols != 32)
		resize(gray, gray, Size(32, 32));

	//cout << "Grayscale image:" << endl << endl << gray << endl << endl;

	FilterEdges(); // find edges
	get_magnitude_hist_block(0, 0); // build histogram for block starting [0,0]

	//Mat hist(1, N_BUCKETS * BLOCK_SIZE * BLOCK_SIZE, CV_64F, block_bucket_vals);
	//cout << "Histogram:" << endl << endl << hist << endl << endl;
	/*cout << "Histogram:" << endl << endl;

	for (int i = 0; i < BLOCK_SIZE * BLOCK_SIZE; i++)
	{
		for (int j = 0; j < N_BUCKETS; j++)
		{
			cout << (j * 20) << " - " << block_bucket_vals[j + i * N_BUCKETS] << " | ";
		}
		cout << endl << endl;
	}*/

	memcpy(input, block_bucket_vals, sizeof(double) * N_BUCKETS * BLOCK_SIZE * BLOCK_SIZE);
}


////////////////////////  NETWORK functions  ///////////////////////////////////

// calculate delta weights from  input to hidden layer
void delta_weights_input_hidden()
{
	for (int i = 0; i < INPUT_SIZE; i++)
	{
		for (int j = 0; j < HIDDEN_SIZE; j++)
		{
			double tmp = 0;
			for (int x = 0; x < OUTPUT_SIZE; x++)
				tmp += (delta[x] * weights_hid_out[j][x]);

			delta_weights_in_hid[i][j] = TETHA * input[i] * hidden[j] * (1 - hidden[j]) * tmp;
		}
	}
}


// calculate delta weights from hidden to output layer
void delta_weights_hidden_output()
{
	for (int j = 0; j < OUTPUT_SIZE; j++)
	{
		delta[j] = (target[j] - output[j])*output[j] * (1 - output[j]);
		for (int i = 0; i < HIDDEN_SIZE; i++)
		{
			delta_weights_hid_out[i][j] = TETHA * delta[j] * hidden[i];
		}
	}
}


// adjust weights
void AdjustWeights()
{
	delta_weights_hidden_output();
	delta_weights_input_hidden();

	for (int j = 0; j < OUTPUT_SIZE; j++)
	{
		for (int i = 0; i < HIDDEN_SIZE; i++)
		{
			weights_hid_out[i][j] += delta_weights_hid_out[i][j];
		}
	}

	for (int i = 0; i < INPUT_SIZE; i++)
	{
		for (int j = 0; j < HIDDEN_SIZE; j++)
		{
			weights_in_hid[i][j] += delta_weights_in_hid[i][j];
		}
	}
}


// set random weights
void SetWeights()
{
	srand((unsigned)time(NULL));

	for (int j = 0; j < INPUT_SIZE; j++)
		for (int i = 0; i < HIDDEN_SIZE; i++)
		{
			weights_in_hid[j][i] = (double)rand() / (RAND_MAX + 1) * (0.5 - (-0.5)) + (-0.5);
		}

	for (int j = 0; j < HIDDEN_SIZE; j++)
		for (int i = 0; i < OUTPUT_SIZE; i++)
		{
			weights_hid_out[j][i] = (double)rand() / (RAND_MAX + 1) * (0.5 - (-0.5)) + (-0.5);
		}
}


// step function implementation
double StepFunction(double param)
{
	return (double)1 / ((double)1 + exp(-param));
}


// multiply input layer by weights from input to hidden layers
void MultInHid()
{
	memset(hidden, 0, sizeof(double) * HIDDEN_SIZE);

	for (int i = 0; i < HIDDEN_SIZE; i++)
	{
		for (int j = 0; j < INPUT_SIZE; j++)
		{
			hidden[i] += input[j] * weights_in_hid[j][i];
		}
		hidden[i] = StepFunction(hidden[i]);
	}
}


// multiply hidden layer by weights from hidden to  output layers
void MultHidOut()
{
	memset(output, 0, sizeof(double) * OUTPUT_SIZE);

	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		for (int j = 0; j < HIDDEN_SIZE; j++)
		{
			output[i] += hidden[j] * weights_hid_out[j][i];
		}
		output[i] = StepFunction(output[i]);
	}
}


// recognize the loaded image
void Recognize()
{
	// multiply input array by input-hidden weights
	MultInHid();

	// multiply hidden array by hidden-output weights
	MultHidOut();

	double max = 0.0;
	int max_ind = -1;
	//cout << endl << endl;
	for (short i = 0; i < OUTPUT_SIZE; i++)
	{
		if (output[i] > max)
		{
			max = output[i];
			max_ind = i;
		}
		//cout << output[i] << " ";
	}
	//cout << endl << endl;
	found_number = max_ind;
	cout << endl << "Recognized number: " << found_number << endl;
}


// performing the single training the network for the current digit 
void Train()
{
	for (short i = 0; i < 10; i++)
		target[i] = 0;

	target[real_number] = 1;

	AdjustWeights();
	Recognize();
}


// performing the continiuos training the network for the current digit 
void ContiniousTrain()
{
	// train the network for the current digit till the network recognizes the digit
	while (1)
	{
		if (real_number == found_number)
			break;
		else
			Train();
	}
}


// statistics data
long train_cycles;	// number cycles is performed during images set training
long img_tested;	// number of images that are tested
long num_errors;	// number of errors (incorrect recognitions)

// training of the images set
void ImagesSetTrainig()
{
	train_cycles = 0;
	while (1)
	{
		img_tested = num_errors = 0;
		for (int i = 0; i <= 9; i++) // for each digit: 0 - 9
		{
			real_number = i;
			for (int j = 0; j < NUM_IMG_FOR_DIGIT_TRAIN; j++) // for each font
			{
				// load image from 'training_images' directory and recognize it
				CHAR Buffer[MAX_PATH];
				char name[20];
				GetCurrentDirectory(256, Buffer);
				strcat(Buffer, "\\training_images\\");
				sprintf(name, "%d_%s.bmp", i, fonts[j].c_str());
				strcat(Buffer, name);
				cout << "Training - " << name << endl;
				String imageName(Buffer);
				LoadImage(imageName);
				Recognize();
				img_tested++;
				if (real_number == found_number) // if correct recognition => continue with the next image
					continue;
				num_errors++;
				ContiniousTrain(); // if incorrect recognition => perform continious training
			}
		}
		train_cycles++;
		if (num_errors == 0)
		{
			// end of training - no errors
			cout << endl << "End of images set training" << endl;
			cout << "Number of cycles: " << train_cycles << endl << endl;
			break;
		}
	}
}

// save weight in file
void SaveWeights()
{
	ofstream fout;
	fout.open("weights.dat", ios::binary);

	for (int i = 0; i < HIDDEN_SIZE; i++)
	{
		for (int j = 0; j < INPUT_SIZE; j++)
		{
			fout.write(reinterpret_cast<char *>(&weights_in_hid[j][i]), sizeof(double));
		}
	}

	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		for (int j = 0; j < HIDDEN_SIZE; j++)
		{
			fout.write(reinterpret_cast<char *>(&weights_hid_out[j][i]), sizeof(double));
		}
	}
	fout.close();
}

// load weight from file
void LoadWeights()
{
	ifstream fin;
	fin.open("weights.dat", ios::binary);

	for (int i = 0; i < HIDDEN_SIZE; i++)
	{
		for (int j = 0; j < INPUT_SIZE; j++)
		{
			fin.read(reinterpret_cast<char *>(&weights_in_hid[j][i]), sizeof(double));
		}
	}

	for (int i = 0; i < OUTPUT_SIZE; i++)
	{
		for (int j = 0; j < HIDDEN_SIZE; j++)
		{
			fin.read(reinterpret_cast<char *>(&weights_hid_out[j][i]), sizeof(double));
		}
	}

	fin.close();
}



int main(int argc, char** argv)
{
	SetWeights();

	char choice = 0;
	while (1)
	{
		cout << endl << endl << "0 - Exit" << endl << "1 - Load image" << endl << "2 - Recognize" 
			 << endl << "3 - One image training" << endl << "4 - Digit to be trained (current: " << real_number << ")" 
			 << endl << "5 - Images set trainig" << endl << "6 - Save weights" << endl << "7 - Load weights" 
			 << endl << "Your choice: ";
		cin >> choice;

		if (choice == '0') // exit the program
			break;

		switch (choice)
		{
			case '1':	// load image from the file 'digit.bmp'
			{
				String imageName("digit.bmp");
				LoadImage(imageName);
				break;
			}
			case '2':	// recognize the loaded image
			{
				Recognize();
				break;
			}
			case '3':	// perform continous training of the current image
			{
				ContiniousTrain();
				break;
			}
			case '4':	// enter the target digit to be trained for the current image
			{
				char digit;
				cout << "Enter the digit to be trained: ";
				cin >> digit;
				real_number = (int)digit - 48;
				break;
			}
			case '5':	// training of the images set 
			{
				ImagesSetTrainig();
				break;
			}
			case '6':	// save weight in file
			{
				SaveWeights();
				break;
			}
			case '7':	// load weight from file
			{
				LoadWeights();
				break;
			}
			default:
				continue;
		}
	}
	
	//waitKey(0); // Wait for a keystroke in the window

	return 0;
}