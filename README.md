## Dehazing using Color-lines
This repository contains an open source implementation of [Fattal's algorithm for dehazing](https://www.cse.huji.ac.il/~raananf/projects/dehaze_cl/). 
It is meant as an accessible implementation with low complexity and ample documentation that can be used as an aid for studying the algorithm.   
  

This is not an efficient implementation; dehazing a single image can take up to a minute on a modern cpu. For an efficient implementation, consider the implementations by [Ekesium](https://github.com/ekexium/dehazing-using-color-lines) and [Tomlk](https://github.com/Tomlk/Dehazing-with-Color-Lines) (same pixel-wise inference, but in scala and C++). Alternatively, consider writing a vectorized implementation. This implementation also does not contain the MRF for interpolation and regularization. Currently linear interpolation is used, which causes artifacts due to missing values (see dark regions in bushes and light edges on bricks).


![Current result](dehaze.gif)

## How to Use 
Use a venv and install the required packages by running `pip install -r requirements.txt`. 
Execute `python dehaze.py` to dehaze `bricks.png`. 


## todos 
* Implement MRF for interpolation and regularization (currently only linear interpolation, which causes artifacts)
* Implement [automatic airlight recovery](https://www.cse.huji.ac.il/~raananf/projects/atm_light/)
