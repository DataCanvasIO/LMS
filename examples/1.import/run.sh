#!/bin/bash
# clean up the model which named opt-125m.
lms del --model_name opt-125m
# import it to LMS.
lms import --model_path ../data/opt-125m
# list model 
lms list
