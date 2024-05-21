CC = gcc
RM = rm -rf
USR_DIR = /home/9yelin9/.local
OMP_DIR = /opt/mpi/gcc-4.8.5/openmpi-4.1.0
GSL_DIR = /opt/gsl
HDF_DIR = /opt/hdf5/intel-2021.4.0/1.12.0
CFLAGS = -g -O2 -Wall -mcmodel=medium -fPIC -I$(USR_DIR)/include -I$(OMP_DIR)/include -I$(GSL_DIR)/include -I$(HDF_DIR)/include -fopenmp
LDFLAGS = -shared -L$(USR_DIR)/lib -L$(OMP_DIR)/lib -L$(GSL_DIR)/lib -L$(HDF_DIR)/lib -fopenmp 
LINKS = -lz -lm -lopenblas -lgsl -lgslcblas -lhdf5 -fPIC
OBJS = mod.o
TARGETS = ../libmod.so

.PHONY: all clean dep
.SUFFIXES : .c .o

.c .o :
	$(CC) $(CFLAGS) -c $<

all : $(TARGETS)
clean :
	$(RM) *.o
	$(RM) $(TARGET)
dep :
	$(CC) $(CFLAGS) -M $(OBJS:.o=.c) 

../libmod.so : mod.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o ../libmod.so mod.o
