.PHONY: all manuscript experiments clean

all: manuscript experiments

manuscript:
	$(MAKE) -C manuscript

experiments:
	$(MAKE) -C experiments

clean:
	$(MAKE) -C manuscript clean
	$(MAKE) -C experiments clean
