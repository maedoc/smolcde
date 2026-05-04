.PHONY: all build test clean install dev

all: build

build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build
	cp build/libsmolmaf.so . 2>/dev/null || true
	cp build/smolcde . 2>/dev/null || true

test: build
	python -m pytest test_cde.py -v

clean:
	cmake --build build --target clean 2>/dev/null || rm -rf build
	rm -f libsmolmaf.so libsmolmaf.dylib smolmaf.dll smolcde smolcde.exe
	rm -f *.maf *.csv

install:
	pip install -e ".[dev]"

dev: install test
