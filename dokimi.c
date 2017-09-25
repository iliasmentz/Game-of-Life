#include "mpi.h"
#include <stdio.h>
#include <math.h>
#define SIZE 16
#define UP    0
#define DOWN  1
#define LEFT  2
#define RIGHT 3

int main(int argc, char *argv[])  {
int numtasks, rank, source, dest, outbuf, i, tag=1,
   inbuf[4]={MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,MPI_PROC_NULL,},
   nbrs[4], dims[2]={4,4},
   periods[2]={1,1}, reorder=0, coords[2];

int sqr = sqrt(SIZE);
MPI_Request reqs[8];
MPI_Status stats[8];
MPI_Comm cartcomm;   // required variable

MPI_Init(&argc,&argv);
MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

if (numtasks == SIZE) {
   // create cartesian virtual topology, get rank, coordinates, neighbor ranks
   MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartcomm);
   MPI_Comm_rank(cartcomm, &rank);

   MPI_Cart_coords(cartcomm, rank, 2, coords);
   //printf("Rank: %d Coords: %d %d\n", rank, coords[0], coords[1]);

   int i, j;
   int neighbor;
   int neighcoord[2];
   if (rank==15){
     for (i= coords[0]-1; (i<=coords[0]+1 ); i++ )
      {
        for (j = coords[1]-1; (j<=coords[1]+1 ); j++)
        {
            if(!(i == coords[0]) || !(j== coords[1]))
            {

              neighcoord[0] = i;
              neighcoord[1] = j;
              MPI_Cart_rank(cartcomm, neighcoord, &neighbor);

              if(coords[0] == neighcoord[0] ){
                if (coords[1] == neighcoord[1]+1)
                  printf("Neighbor is %d, left\n", neighbor);
                else
                  printf("Neighbor is %d, right\n", neighbor);
              }
              else if ( coords[0] == neighcoord[0]+1 ) {
                if ( coords[1] == neighcoord[1]+1 )
                  printf("Neighbor is %d, Up left\n", neighbor);
                else if ( coords[1] == neighcoord[1])
                  printf("Neighbor is %d, Up \n", neighbor);
                else
                  printf("Neighbor is %d, Up right\n", neighbor);
              }
              else if( coords[0] == neighcoord[0]-1 ) {
                if ( coords[1] == neighcoord[1]+1)
                  printf("Neighbor is %d, Down left\n", neighbor);
                else if ( coords[1] == neighcoord[1] )
                  printf("Neighbor is %d, Down\n", neighbor);
                else
                  printf("Neighbor is %d, Down right\n", neighbor);
              }
            }
        }
      }
   }

}
else
   printf("Must specify %d processors. Terminating.\n",SIZE);

MPI_Finalize();
return 0;
}
