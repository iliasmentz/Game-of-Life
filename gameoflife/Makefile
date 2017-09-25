#
# Variables and Flags
#

CC        = mpicc
CFLAGS   += -DMPI


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
	$(CC) -o $(PROGRAM) $(SRCS) $(CFLAGS) $(LIBS)

clean:
	/bin/rm -f $(OBJS) $(PROGRAM)
