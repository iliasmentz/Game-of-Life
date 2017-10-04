#ifndef BCCD_LIFE_H
#define BCCD_LIFE_H

#include <mpi.h>

#include "Defaults.h" // For Life's constants

#include <time.h>     // For seeding random
#include <stdlib.h>   // For malloc et al.
#include <stdbool.h>  // For true/false
#include <getopt.h>   // For argument processing
#include <stdio.h>    // For file i/o
#include <string.h>
#include <unistd.h>

int               init (struct life_t * life, int * c, char *** v);
void        eval_rules (struct life_t * life);
void       copy_bounds (struct life_t * life);
int       update_grid (struct life_t * life);
void    allocate_grids (struct life_t * life);
void        init_grids (struct life_t * life);
void        write_grid (struct life_t * life);
void        free_grids (struct life_t * life);
double     rand_double ();
void    randomize_grid (struct life_t * life, double prob);
void       seed_random (int rank);
void           cleanup (struct life_t * life);
void        parse_args (struct life_t * life, int argc, char ** argv);
void             usage ();


#endif
