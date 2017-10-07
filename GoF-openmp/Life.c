/*******************************************
To run:
./Life [Rows] [Columns] [Generations] [Display]

See the README included in this directory for
more detailed information.
*******************************************/

#include "Life.h"
#include "Defaults.h" // For Life's constants

int main(int argc, char ** argv) {
	int count;
	struct life_t life;
	int result;
	int change;
	init(&life, &argc, &argv);
	double t1, t2;
	t1 = MPI_Wtime();
	for (count = 0; count < life.generations; count++) {
		//printf("Generation %d\n", count );
		copy_bounds(&life);

		result = eval_rules(&life);
		update_grid(&life);
		MPI_Allreduce(&change, &result,1,MPI_INT,MPI_SUM, life.comm);

		MPI_Barrier(life.comm);
		if(result == 0)
			break;
	}
	t2 = MPI_Wtime();
	printf("Rank %d elapsed time is %.2f\n", life.rank, t2 - t1);

	cleanup(&life);


	exit(EXIT_SUCCESS);
}
