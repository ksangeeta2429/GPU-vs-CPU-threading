/*********************************************************************************
 ** FILENAME        :  kumari_sangeeta_persistent.c
 **
 ** DESCRIPTION     :  This file defines the functions necessary to implement the convergence of
 **					   the dissipation model using persistent threads                  
 ** Revision History:
 ** DATE           NAME                REASON
 ** ------------------------------------------------------------------------------
 ** March 8 2016   Sangeeta Kumari     Parallel Computing Project
 ********************************************************************************/

  /*********************************************************************************
 ** HEADER FILES
 ********************************************************************************/
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <ctime>
#include <chrono>
#include <vector>
#include <sstream>
#include <omp.h>

using namespace std;
using namespace std::chrono;

#define MAX_ITERATIONS 1000000

 /*********************************************************************************
 ** GLOBAL VARIABLES
 ********************************************************************************/
struct box
{
	int boxId;
	int xLeft, xRight;
	int yLeft, yBottom;
	int height;
	int width;
	int topN;
	int bottomN;
	int leftN;
	int rightN;
	vector <int> topNeighbors;
	vector <int> bottomNeighbors;
	vector <int> leftNeighbors;
	vector <int> rightNeighbors;
	int perimeter;
	double temp;
};

vector<box> matrix;
vector<double> temporary;
int no_of_threads;
int no_boxes_g;
int stop_iter = 0;
int iterations_g = 0;
double affect_rate, epsilon;
pthread_barrier_t barrier, loopBarrier, convergenceBarrier;

 /*********************************************************************************
 ** FUNCTION DECLARATIONS
 ********************************************************************************/
void parseFile(vector<box>& matrix);
void printMatrix(vector<box>& matrix);
void printBox(box block);
void convergence(vector<box>& matrix, double epsilon, double affect_rate);
double temperatureDiff(int id, vector<box>& matrix);
double tempOverPerimeter(box block, vector<box>& matrix);
int contactDist(int id1, int id2, vector<box>& matrix);
void calPerimeter(vector<box>& matrix);
void *calc_update(void *);
bool testConvergence();

/*********************************************************************************
 ** FUNCTION NAME : main()
 **
 ** DESCRIPTION   : It calls the appropriate functions for the implementation 
 **
 ** RETURNS       : 0 if successfully executed else exits with value 0
 ********************************************************************************/

int main(int argc, char *argv[])
{
    system_clock::time_point t1 = system_clock::now();
	int noBoxes,noRows,noCols;	
	string line;
	int id;

    if(argc < 4)
	{
		cout<<"\nExecute like ./a.out <AFFECT RATE> <EPSILON> < test_file\n";
		exit(0);
	}

	affect_rate = atof(argv[1]);
	epsilon = atof(argv[2]);
	no_of_threads = atoi(argv[3]);

	parseFile(matrix);
	calPerimeter(matrix);
	//printMatrix(matrix);
	clock_t begin = clock();
	time_t start = time(NULL);
    system_clock::time_point start1 = system_clock::now();
	convergence(matrix, epsilon, affect_rate);
	clock_t end = clock();
	time_t finish = time(NULL);
    system_clock::time_point end1 = system_clock::now();
  	double elapsedSecs = double(end - begin)/CLOCKS_PER_SEC;
  	double elapsedTime = double(finish - start);
  	cout<<"\nelapsed convergence loop time (clock): "<<elapsedSecs;
  	cout<<"\nelapsed convergence loop time  (time): "<<elapsedTime;
    cout <<"\nelapsed convergence loop time (chrono): "<<std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
	cout<<endl<<"***********************************************************"<<endl;
	return 0;
}

/*********************************************************************************
 ** FUNCTION NAME : parseFile()
 **
 ** DESCRIPTION   : parses the input file and saves the value in the struct 
 **
 ** RETURNS       : NIL
 ********************************************************************************/
void parseFile(vector<box>& matrix)
{
	int noBoxes, noRows, noCols;
	int id;	
	string line = "";
	if (true)
	{
	    cin>>noBoxes;
	    no_boxes_g = noBoxes;
	    cin>>noRows;
	    cin>>noCols;

		for (int i=0; i < noBoxes; i++)
		{
		  	matrix.push_back(box());
		  	cin>>matrix[i].boxId;

		  	cin>>matrix[i].xLeft;
		  	cin>>matrix[i].yLeft;
		  	cin>>matrix[i].height;
		  	cin>>matrix[i].width;

		  	
		  	matrix[i].xRight = matrix[i].xLeft + matrix[i].width;
		  	matrix[i].yBottom = matrix[i].yLeft + matrix[i].height;

		  	cin>>matrix[i].topN;

		  	for (int j=0; j<matrix[i].topN; j++)
		  	{
		  		cin>>id;
		  		matrix[i].topNeighbors.push_back(id);
			}

		  	cin>>matrix[i].bottomN;
		  	for (int j=0; j<matrix[i].bottomN; j++)
		  	{
		  		cin>>id;
		  		matrix[i].bottomNeighbors.push_back(id);
			}

		  	cin>>matrix[i].leftN;
		  	for (int j=0; j<matrix[i].leftN; j++)
		  	{
		  		cin>>id;
		  		matrix[i].leftNeighbors.push_back(id);
			}

		  	cin>>matrix[i].rightN;
		  	for (int j=0; j<matrix[i].rightN; j++)
		  	{
		  		cin >> id;
		  		matrix[i].rightNeighbors.push_back(id);
			}
			cin>>matrix[i].temp;
		}
	}
}

/*********************************************************************************
 ** FUNCTION NAME : convergence()
 **
 ** DESCRIPTION   : helps the data to converge using the condition (max-min) < (max*epsilon)
 **
 ** RETURNS       : NIL
 ********************************************************************************/
void convergence(vector<box>& matrix , double epsilon, double affect_rate)
{
	vector<double> temporary;
	int j=0;
	int noThreads_actual=0;
	double min, max, diff;

	for(int i=0; i<matrix.size(); i++) {
		temporary.push_back(0);
	}

   	omp_set_num_threads(no_of_threads);
	#pragma omp parallel 
	{
		while(stop_iter != 1){

			#pragma omp for private(diff)
			for(int i=0; i<matrix.size(); i++)
			{
				/* calls temperatureDiff to update the current temp of the box taking into account the temp of the neighbors */
				noThreads_actual = omp_get_num_threads();
				diff = temperatureDiff(i, matrix);
				temporary[i] = matrix[i].temp - diff*affect_rate;
			}

			#pragma omp barrier

			#pragma omp master
			{
				min = temporary[0];
				max = temporary[0];
				for(int i=0; i<matrix.size(); i++)
				{
					matrix[i].temp = temporary[i];
					if(temporary[i] < min)
					{
						min = temporary[i];
					}
					if(temporary[i] > max)
					{
						max = temporary[i];
					}
				}

				j++;
				
				/* the iteration stops when the program converges */
				if((max-min) < (max*epsilon))
				{
					stop_iter = 1;
				} 
			}
		
			#pragma omp barrier
    	}
    }
    cout<<endl<<"***********************************************************"<<endl;
	cout<<"Dissipation converged in: "<<j<<" iterations,"<<endl<<" with max DSV = "<< max<<" and min DSV = "<<min<<endl;
	cout<<"\t affect rate = "<<affect_rate<<"       epsilon= "<<epsilon<<endl;
	cout<<"\nActual no of threads used:"<<noThreads_actual<<endl;
}


void calc_update(){

}
/*********************************************************************************
 ** FUNCTION NAME : temperatureDiff()
 **
 ** DESCRIPTION   : calculates the temp difference by:
 **					(temp of current box - weighted average adjacent temperature)
 **
 ** RETURNS       : NIL
 ********************************************************************************/
double temperatureDiff(int id, vector<box>& matrix)
{
	double diff;
	double tempN = tempOverPerimeter(matrix[id],matrix);
	diff = matrix[id].temp - (tempN/matrix[id].perimeter);
	return diff;
}

/*********************************************************************************
 ** FUNCTION NAME : tempOverPerimeter()
 **
 ** DESCRIPTION   : finds the weighted average of a box 
 **
 ** RETURNS       : temperature
 ********************************************************************************/
double tempOverPerimeter(box block, vector<box>& matrix)
{
	double temperature = 0.0;
	int id;

	for (int i=0; i<block.topN; i++)
	{
		id = block.topNeighbors[i];
		temperature += contactDist(block.boxId,id,matrix) * matrix[id].temp;
	}

	for (int i=0; i<block.bottomN; i++)
	{
		id = block.bottomNeighbors[i];
		temperature += contactDist(block.boxId,id,matrix) * matrix[id].temp;
	}

	for (int i=0; i<block.leftN; i++)
	{
		id = block.leftNeighbors[i];
		temperature += contactDist(block.boxId,id,matrix) * matrix[id].temp;
	}

	for (int i=0; i<block.rightN; i++)
	{
		id = block.rightNeighbors[i];
		temperature += contactDist(block.boxId,id,matrix) * matrix[id].temp;
	}
	return temperature; 
}

/*********************************************************************************
 ** FUNCTION NAME : contactDist()
 **
 ** DESCRIPTION   : finds the overall contact distance of a box with its neighbors
 **
 ** RETURNS       : dist
 ********************************************************************************/
int contactDist(int id1, int id2, vector<box>& matrix)
{
	int dist = 0;
	if((matrix[id1].xLeft <= matrix[id2].xLeft) && (matrix[id2].xLeft < matrix[id1].xRight))
	{
		if(matrix[id2].xRight < matrix[id1].xRight)
		{
			dist =  matrix[id2].width;
		}
		else
		{
			dist =  matrix[id1].xLeft + matrix[id1].width - matrix[id2].xLeft;
		}
	}
	else if((matrix[id2].xLeft <= matrix[id1].xLeft ) && (matrix[id1].xLeft < matrix[id2].xRight))
	{
		if(matrix[id1].xRight < matrix[id2].xRight)
		{
			dist = matrix[id1].width;
		}
		else
		{
			dist = matrix[id2].xRight - matrix[id1].xLeft;
		}
	}
	else if((matrix[id1].yLeft <= matrix[id2].yLeft) && (matrix[id2].yLeft < matrix[id1].yBottom))
	{
		if(matrix[id2].yBottom < matrix[id1].yBottom)
		{
			dist = matrix[id2].height;
		}
		else
		{
			dist = matrix[id1].yBottom - matrix[id2].yLeft;
		}
	}
	else if((matrix[id2].yLeft <= matrix[id1].yLeft) && (matrix[id1].yLeft < matrix[id2].yBottom))
	{
		if(matrix[id1].yBottom < matrix[id2].yBottom)
		{
			dist = matrix[id1].height;
		}
		else
		{
			dist = matrix[id2].yBottom - matrix[id1].yLeft;
		}
	}
	return dist;
}

/*********************************************************************************
 ** FUNCTION NAME : calPerimeter()
 **
 ** DESCRIPTION   : calculates the perimeter of the box by using the aggregate 
 **					contact distance with the neighbors 
 **
 ** RETURNS       : NIL
 ********************************************************************************/
void calPerimeter(vector<box>& matrix)
{
	int id;

	for(int i=0; i<matrix.size(); i++)
	{
		int perimeter = 0;

		for (int j=0; j<matrix[i].topN; j++)
		{
			id = matrix[i].topNeighbors[j];
			perimeter += contactDist(matrix[i].boxId, id, matrix);
		}
		for (int j=0; j<matrix[i].bottomN; j++)
		{
			id = matrix[i].bottomNeighbors[j];
			perimeter += contactDist(matrix[i].boxId, id, matrix);
		}

		for (int j=0; j<matrix[i].leftN; j++)
		{
			id = matrix[i].leftNeighbors[j];
			perimeter += contactDist(matrix[i].boxId, id, matrix);
		}

		for (int j=0; j<matrix[i].rightN; j++)
		{
			id = matrix[i].rightNeighbors[j];
			perimeter += contactDist(matrix[i].boxId, id, matrix);
		}
		matrix[i].perimeter = perimeter;
	}
}
