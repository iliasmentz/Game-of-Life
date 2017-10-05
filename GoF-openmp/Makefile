#
# Variables and Flags
#

CC        = mpicc
CFLAGS   += -fopenmp


LIBS     += -lm
CFLAGS   +=


PROGRAM   = Life
SRCS      = Life.c  LifeFunctions.c
OBJS      = $(SRCS:.c=.o)		# object file

#
# Targets
#

default: all

all: $(PROGRAM)

$(PROGRAM): $(OBJS)
	$(CC) $(CFLAGS) -o $(PROGRAM) $(SRCS) $(LIBS)

clean:
	/bin/rm -f $(OBJS) $(PROGRAM)
