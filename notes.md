

Schedule:

1. Algorithm in the paper - how to update all the points based on the candidate set obtained from algorithm (comparing lower and upper subsystem) (Mar 10)
2. Implementation: 
  - Use the threshold defined in leaking_corner.py
  - What are the metrics?
  - Finish 2d case for intersection and provide the results. (Mar 7 EOD)
    - Computation time
    - Number of points (refer line 146 to end in local_update_numpy.py)
3. 6D example (Mar )
4. 




Schedule:
1. (Mugilan - Mar 7) Implementation for the 2d case for intersection (with metrics).
2. (Mugilan - Mar 10) Algorithm 2 - to update all the points based on the candidate set obtained from algorithm 1 in paper.
3. (Chong - Mar 8) 6D example.
4. (Mugilan - March 11) 6D example implementation.



- Add per-iter stats - done
- Add time stats - done
- Add error stats - done


Algorithm:
- Add explanation text for algorithm
- Modify algorithm to handle 

Implementation:
- Optimization to avoid recomputing indices which have already been computed
- Delta as a command-line parameter
- New algorithm:
  - Find the indices only for first iteration. 
  - Just use the updated vertices for all future iterations.

Experiments:
- Varying delta.

Writing:
- Observations on the experiments
  - Difference b/w old delta formula and new one - Even with different deltas, we still get exactly correct results.  
  - Quantifying work done: 
    - Include work done by decomposition method. 
    - But it is not the same as it is on 1D.
    - Show how work done corresponds to execution time
  - 



