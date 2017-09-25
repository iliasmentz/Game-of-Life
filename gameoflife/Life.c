/*******************************************
A C implementation of Conway's Game of Life.

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

	init(&life, &argc, &argv);

	for (count = 0; count < life.generations; count++) {

		copy_bounds(&life);

		eval_rules(&life);

		update_grid(&life);
		MPI_Barrier(life.comm);
	}

	cleanup(&life);

	exit(EXIT_SUCCESS);
}
