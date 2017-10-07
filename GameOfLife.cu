#include <stdio.h>
#include <stdlib.h>
#include <time.h>



#define BLOCK_SIZE 256

__global__ void rows (int dimension , int *array)
{
    //theloume to id na anhkei sto diasthma [1,dimension]
    int id = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (id < dimension)
    {
        // antigrafw thn prwth grammh tou pinaka sthn teleutaia grammh tou voi8itikou
        array[(dimension+2)*(dimension+1) + id] = array[(dimension+2)+id];

        //antigrafw thn teleutaia grammh tou pinaka sthn prwth grammh tou voi8itikou
        array[id] = array[(dimension+2)*dimension + id];

    }

}

__global__ void columns (int dimension , int *array)
{

    //theloume to id na anhkei sto diasthma [0 , dimension+1]
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id <= dimension+1)
    {

    // antigrafw thn prwth sthlh tou pinaka sthn teleutaia sthlh tou voi8itikou
    array[id*(dimension+2)+dimension+1] = array[id*(dimension+2)+1];
    //antigrafw thn teleutaia sthlh tou pinaka sthn prwth sthlh tou voi8itikou
    array[id*(dimension+2)] = array[id*(dimension+2) + dimension];

    }

}

__global__ void GameOfLife (int dimension , int *array , int *helparray)
{ 

    //theloume to id na anhkei sto diasthma [0 , dimension+1]
    int iy = blockDim.y * blockIdx.y + threadIdx.y +1;
    int ix = blockDim.x * blockIdx.x +threadIdx.x +1;
    int id = iy * (dimension+2) + ix;

    int aliveNeighbors;

    if (iy <= dimension && ix <= dimension)
    {

      //edw pernoume ton ari8mo twn geitwnwn gia ena sugkekrimeno shmeio ston pinaka
      aliveNeighbors = array[id+(dimension+2)] + array[id-(dimension+2)] //upper lower
                      + array[id+1] + array[id-1]             //right left
                      + array[id+(dimension+3)] + array[id-(dimension+3)] //diagonals
                      + array[id-(dimension+1)] + array[id+(dimension+1)];


      int currentCell = array[id]; //to kuttaro gia to opoio 8eloume na vroume gia to an 8a epiviwsei h oxi

                      //efarmogh twn kanonwn tou paixnidiou panw sto kuttaro

     if (aliveNeighbors < 2 || aliveNeighbors > 3)
      		helparray[id] = 0;  // den epibiwnei logo uperplh8ismouy
      else if(currentCell != 0 || aliveNeighbors == 3)
      		helparray[id] = 1;
      else 
      		helparray[id] = currentCell;
  }
}

int main (int argc , char* argv[] )
{
    int i ,j , k , linDimension , genies , survivedCells=0;

    int *device_pinakas, *device_new_pinakas , *device_temp_pinakas, *host_pinakas , *host_temp , *host_temp1 , l , counter;//oi pinakes pou ksekinane me device einai autoi pou tha treksoun sthn GPU enw o host tha treksei sthn CPU

    linDimension = 9600;//atoi(argv[1]);
    genies =200;

    host_pinakas = (int*)malloc(sizeof(int)*(linDimension+2)*(linDimension+2));
    cudaMalloc(&device_pinakas , sizeof(int)*(linDimension+2)*(linDimension+2));   //sth cuda h desmeush mnhmhs ginetai me thn cudaMalloc
    cudaMalloc(&device_new_pinakas , sizeof(int)*(linDimension+2)*(linDimension+2));

    //arxikopoiw ton plh8hsmo tuxaia

    srand(time(NULL));
    for (i = 0; i <= linDimension; i++){
      for (j = 0; j <= linDimension;  j++){
          host_pinakas[i*(linDimension+2) + j] = rand() % 2;
      }
    }

   


    cudaMemcpy(device_pinakas , host_pinakas , sizeof(int)*(linDimension+2)*(linDimension+2) , cudaMemcpyHostToDevice); //antigrafw ston device pinaka ton host pinanaka.. to 4o orisma dhlwnei ton tupo ths antigrafhs

    /*dim3 blocks(BLOCK_SIZE,1,1);
    dim3 blocksize(BLOCK_SIZE ,BLOCK_SIZE ,1);
    int linear_pin = linDimension/BLOCK_SIZE +1;
    dim3 gridsize(linear_pin,linear_pin,1);
    dim3 rsize((int)ceil(linDimension/(float)blocks.x),1,1); //megethos grammwn r-rows
    dim3 csize((int)ceil((linDimension+2)/(float)blocks.x),1,1); //megethos sthlwn c-columns*/

    int blocksize = 256;
    int numBlocks = (linDimension +blocksize -1)/blocksize;
    //kentriki epanalhpsh tou paixnidiou
    for (k = 0; k <= genies; k++)
    {	
    	

        host_temp = (int*)malloc(sizeof(int)*(linDimension+2)*(linDimension+2));
        host_temp1= (int*)malloc(sizeof(int)*(linDimension+2)*(linDimension+2));
        rows<<< numBlocks , 256>>>(linDimension, device_pinakas);
        columns<<<numBlocks , 256>>>(linDimension , device_pinakas);
        GameOfLife<<<numBlocks , 256>>>(linDimension, device_pinakas , device_new_pinakas);
        cudaMemcpy(host_temp , device_pinakas ,sizeof(int)*(linDimension+2)*(linDimension+2) ,cudaMemcpyDeviceToHost);
        cudaMemcpy(host_temp1 , device_pinakas ,sizeof(int)*(linDimension+2)*(linDimension+2) , cudaMemcpyDeviceToHost);
        
        //ret = memcpy(device_pinakas , device_new_pinakas ,sizeof(device_pinakas));

        counter=0;
        for (l =1; l <= linDimension; l++)
        {
           if(host_temp[l] == host_temp1[l])
          counter++;
        }        

        device_temp_pinakas =  device_pinakas;
        device_pinakas =device_new_pinakas;
        device_new_pinakas = device_temp_pinakas;

        free(host_temp);
        free(host_temp1);
        if (counter == linDimension)
            break;
        


    }

    cudaMemcpy(host_pinakas , device_pinakas , sizeof(int)*(linDimension+2)*(linDimension+2) , cudaMemcpyDeviceToHost);

    for (i = 1; i<=linDimension; i++){
      for(j = 1; j<=linDimension; j++){
        survivedCells += host_pinakas[i*(linDimension+2)+j];
      }

    }

    printf(" Epiviwsan sunolika %d kuttara\n", survivedCells );
    free(host_pinakas);
    cudaFree(device_pinakas);
    cudaFree(device_new_pinakas);

}
