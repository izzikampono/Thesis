#!/usr/bin/env python
# template from documentation
 
import multiprocessing 
import os # For reading the number of CPUs requested. 
import time # For clocking the calculation. 
 
def double(data):
    return data * 2
 
if __name__ == '__main__':
    begin = time.time()
    inputs = list(range(10)) # Makes an array from 0 to 10
    poolSize = int(os.environ['SLURM_JOB_CPUS_PER_NODE']) # Number of CPUs requested.
    pool = multiprocessing.Pool(processes=poolSize,)
    poolResults = pool.map(double, inputs) # Do the calculation.
    pool.close() # Stop pool accordingly.
    pool.join()  # Wrap up data from the workers in the pool.
    print ('Pool output:', poolResults) # Results.
    elapsedTime = time.time() - begin
    print ('Time elapsed for ' , poolSize, ' workers: ', elapsedTime, ' seconds')