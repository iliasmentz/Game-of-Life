#ifndef DEFAULTS_H
#define DEFAULTS_H

#include <stdbool.h>
#include <getopt.h>

static const char * opts = "c:r:g:i:o::h?";
static const struct option long_opts[] = {
	{ "columns", required_argument, 0, 'c' },
	{ "rows", required_argument, 0, 'r' },
	{ "gens", required_argument, 0, 'g' },
	{ "output", required_argument, 0, 'o' },
	{ "input", required_argument, 0, 'i' },
	{ "help", no_argument, 0, 'h' },
	{ "", no_argument, 0, 0 }
};


// All the data needed by an instance of Life
struct life_t {
	int  rank;
	int  size;
	int  ncols;
	int  nrows;
	int  ** grid;
	int  ** next_grid;
	int  generations;
	char * infile;
	char * outfile;
};

enum CELL_STATES {
	DEAD = 0,
	ALIVE
};



#endif
