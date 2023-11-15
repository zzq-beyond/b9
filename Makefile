TARGET = pro

SRC_DIR = ./src/cpp
SRC_SUBDIR += . 
INCLUDE_DIR += .
OBJ_DIR = ./obj

CC = g++
C_FLAGS = -g -Wall
LD = $(CC)

INCLUDES += -I$(INCLUDE_DIR) \
            -I/usr/local/include/opencv4 \
			-I/opt/boost/include \
			-I/opt/intel/openvino_2022/runtime/include
			


LD_FLAGS += -L/usr/local/lib \
			-L/opt/boost/lib \
			-L/opt/intel/openvino_2022/runtime/lib/intel64
			

LD_LIBS += -lopencv_core \
           -lopencv_highgui \
           -lopencv_imgcodecs \
           -lopencv_imgproc \
		   -lboost_filesystem \
		   -lboost_json \
		   -lopenvino
		   
		   



ifeq ($(CC), g++)
	TYPE = cpp
else
	TYPE = c
endif

SRCS += ${foreach subdir, $(SRC_SUBDIR), ${wildcard $(SRC_DIR)/$(subdir)/*.$(TYPE)}}
OBJS += ${foreach src, $(notdir $(SRCS)), ${patsubst %.$(TYPE), $(OBJ_DIR)/%.o, $(src)}}

vpath %.$(TYPE) $(sort $(dir $(SRCS)))

all : $(TARGET)
	@echo "Builded target:" $^
	@echo "Done.\n\n"

run : all
	@./pro

$(TARGET) : $(OBJS)
	@mkdir -p $(@D)
	@echo "Linking" $@ "from" $^ "..."
	$(LD) -o $@ $^ $(LD_FLAGS) $(LD_LIBS)
	@echo "Link finished\n"

$(OBJS) : $(OBJ_DIR)/%.o:%.$(TYPE)
	@mkdir -p $(@D)
	@echo "Compiling" $@ "from" $< "..."
	$(CC) -c -o $@ $< $(C_FLAGS) $(INCLUDES)
	@echo "Compile finished\n"

.PHONY : clean cleanobj
clean : cleanobj
	@echo "Remove all executable files"
	rm -f $(TARGET)
cleanobj :
	@echo "Remove object files"
	rm -rf $(OBJ_DIR)/*.o
