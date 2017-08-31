#
# Variables and Flags
#

CC        = mpicc
CFLAGS   += -DMPI


LIBS     += -lm
CFLAGS   += -fopenmp


PROGRAM   = Life
SRCS      = Life.c  LifeFunctions.c
OBJS      = $(SRCS:.c=.o)		# object file

#
# Targets
#

default: all

all: $(PROGRAM)

$(PROGRAM): $(OBJS)
	$(CC) -o $(PROGRAM) $(SRCS) $(CFLAGS)

clean:
	/bin/rm -f $(OBJS) $(PROGRAM)
