# ml PDC/22.06 rocm/5.0.2 craype-accel-amd-gfx90a anaconda3

SRC_DIR		= ./src
APP_DIR         = ./apps

DEPS 		= $(SRC_DIR)/Pop.h $(SRC_DIR)/Prj.h $(SRC_DIR)/Globals.h $(SRC_DIR)/Pats.h $(SRC_DIR)/Parseparam.h $(SRC_DIR)/Logger.h
OBJS 		= $(SRC_DIR)/Pop.o $(SRC_DIR)/Prj.o $(SRC_DIR)/Globals.o $(SRC_DIR)/Pats.o $(SRC_DIR)/Parseparam.o $(SRC_DIR)/Logger.o

CXX		= g++
MPICXX		= mpic++ 

INCLUDE		= -I$(SRC_DIR) -I/path/to/mpi/include # for header files

FLAGS		= -O0 -g
MPIXX_FLAGS	= -lmpi ${PE_MPICH_GTL_LIBS_amd_gfx90a} -lopenblas

%.o: %.cpp $(DEPS)
	$(MPICXX) -c -o $@ $< $(INCLUDE) $(FLAGS) 

reprlearn: $(APP_DIR)/reprlearn/reprlearnmain.o $(OBJS)
	$(MPICXX) -o $(APP_DIR)/reprlearn/reprlearnmain $^ $(INCLUDE) $(FLAGS) $(MPIXX_FLAGS)

all: clean reprlearn

.PHONY: clean reprlearn all
clean : 
	rm -f *.o *.bin *.log *.png *.gif *.out out.txt err.txt *~ core reprlearnmain
	rm -f $(SRC_DIR)/*.o $(SRC_DIR)/*.bin $(SRC_DIR)/*~
	rm -f $(APP_DIR)/reprlearn/*.o $(APP_DIR)/reprlearn/*~ $(APP_DIR)/reprlearn/reprlearnmain
