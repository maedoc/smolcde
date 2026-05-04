.PHONY: all build test clean install dev lint

PYTHON ?= python3

all: build

build:
	cmake -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build
	-cp build/libsmolmaf.so . 2>/dev/null
	-cp build/smolcde . 2>/dev/null

test: build
	$(PYTHON) -m pytest test_cde.py -v -n auto

clean:
	-cmake --build build --target clean 2>/dev/null || rm -rf build
	-rm -f libsmolmaf.so libsmolmaf.dylib smolmaf.dll smolcde smolcde.exe
	-rm -f *.maf *.csv

install:
	pip install -e ".[dev]"

dev: install test

lint:
	$(PYTHON) -m ruff check cde.py test_cde.py
	$(PYTHON) -m ruff format --check cde.py test_cde.py
