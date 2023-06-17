./setup.py build_ext --inplace

dockertest:
   make setup
   make build
   make run
