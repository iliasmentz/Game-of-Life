#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "timer.h"

#define BLOCKSIZE 32


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

int main ()
{
    int i ,j , k , linDimension , genies , survivedCells=0;
    double elapsed , start , finished;

    int *device_pinakas, *device_new_pinakas , *device_temp_pinakas, *host_pinakas , *host_temp , *host_temp1 , l , counter;//oi pinakes pou ksekinane me device einai autoi pou tha treksoun sthn GPU enw o host tha treksei sthn CPU

    linDimension =600; // megethos pinaka (pinakas linDimensionxlinDimension)
    genies =200;

    host_pinakas = (int*)malloc(sizeof(int)*(linDimension+2)*(linDimension+2));
    cudaMalloc(&device_pinakas , sizeof(int)*(linDimension+2)*(linDimension+2));   //sth cuda h desmeush mnhmhs ginetai me thn cudaMalloc
    cudaMalloc(&device_new_pinakas , sizeof(int)*(linDimension+2)*(linDimension+2));


    srand(time(NULL));
    for (i = 0; i <= linDimension; i++){
      for (j = 0; j <= linDimension;  j++){                             //arxikopoiw ton plh8hsmo tuxaia
          host_pinakas[i*(linDimension+2) + j] = rand() % 2;
      }
    }

   


    cudaMemcpy(device_pinakas , host_pinakas , sizeof(int)*(linDimension+2)*(linDimension+2) , cudaMemcpyHostToDevice); //antigrafw ston device pinaka ton host pinanaka.. to 4o orisma dhlwnei ton tupo ths antigrafhs

   

   
    dim3 blocksize(BLOCKSIZE , BLOCKSIZE ,1); // gia ton GOL kernel 
    int lineargrid = linDimension/BLOCKSIZE +1;
    dim3 gridSize(lineargrid , lineargrid , 1);  //blocks pou 8a exei to GOL kernel .. ousiastika einai sunolo keliwn tou pinaka dia ta threads (opou threads=BLOCKSIZE*BLOCKSIZE)

    dim3 copyBLOCKSIZE(BLOCKSIZE , 1 ,1);                //kanw define to BLOCKSIZE iso me th riza twn threads dld 16 gia 256 kai 32 gia 1024 gia na to kalesw stous kernels rows kai columns pou xrhsimopoioyn mia diastash
    dim3 gridRows((int)ceil(linDimension/(float)copyBLOCKSIZE.x),1,1);   //upologismos blocks gia ton rows kernels.. diairw to sunolo twn grammwn me to BLOCKSIZE
    dim3 gridColumns((int)ceil((linDimension+2)/(float)copyBLOCKSIZE.x),1,1); //upologismos blocks gia ton colums kernel.. diairw to sunolo twn sthlwn+2  me to BLOCKSIZE

    GET_TIME(start);

    //kentriki epanalhpsh tou paixnidiou
    for (k = 0; k <= genies; k++)
    {	
    	

        host_temp = (int*)malloc(sizeof(int)*(linDimension+2)*(linDimension+2));      // gia na elegxw an meta apo mia gennia den ginetai allagh ftiaxnw duo pinakes pou se autous antigrafw tous device pinakes
        host_temp1= (int*)malloc(sizeof(int)*(linDimension+2)*(linDimension+2));        //pou se autous antistoixoun h prohgoumenh kai epomenh genia , tous sugkrinw kai meta apodesmeuw 
        rows<<< gridRows , copyBLOCKSIZE>>>(linDimension, device_pinakas);
        columns<<<gridColumns , copyBLOCKSIZE>>>(linDimension , device_pinakas);
        GameOfLife<<<gridSize , blocksize>>>(linDimension, device_pinakas , device_new_pinakas);
        cudaMemcpy(host_temp , device_pinakas ,sizeof(int)*(linDimension+2)*(linDimension+2) ,cudaMemcpyDeviceToHost);
        cudaMemcpy(host_temp1 , device_pinakas ,sizeof(int)*(linDimension+2)*(linDimension+2) , cudaMemcpyDeviceToHost);


        
       
        counter=0;
        for (l =1; l <= linDimension; l++)                              //elegxw tous duo pinakes diladh ths epomenhs kai prohgoumenhs genias gia na dw an exoun allaksei
        {
            //printf("%d kai new %d\n",host_temp[l] , host_temp1[l] );
           if(host_temp[l] == host_temp1[l])
          counter++;
        }        

        device_temp_pinakas =  device_pinakas;                   //afou perasei mia genia allazw tis times etsi wste sthn epomenh epanalhpsh h new genia na einai h palia
        device_pinakas =device_new_pinakas;
        device_new_pinakas = device_temp_pinakas;

        free(host_temp);
        free(host_temp1);
        if (counter == linDimension)                              //an exoun paramenei idioi kanw break thn epanalhpsh kai apodesmeuw tous pinakes
            break;
        


    }

    cudaMemcpy(host_pinakas , device_pinakas , sizeof(int)*(linDimension+2)*(linDimension+2) , cudaMemcpyDeviceToHost);

    for (i = 1; i<=linDimension; i++){
      for(j = 1; j<=linDimension; j++){
        survivedCells += host_pinakas[i*(linDimension+2)+j];    //euresh twn epizwntwn kuttarwn
      }

    }

    GET_TIME(finished);
    elapsed = finished - start;
    printf("the code executed in  %.8f secs\n",elapsed);

    printf(" Epiviwsan sunolika %d kuttara\n", survivedCells );
    free(host_pinakas);
    cudaFree(device_pinakas);
    cudaFree(device_new_pinakas);

}
