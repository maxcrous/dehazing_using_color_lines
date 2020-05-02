## Dehazing using Color-lines
This repository contains an open source implementation of [Fattal's algorithm for dehazing](https://www.cse.huji.ac.il/~raananf/projects/dehaze_cl/). 
It is meant as an accessible implementation with low complexity and ample documentation that can be used as an aid for studying the algorithm.   
  
For a compiled implementation, consider the implementations by [Ekesium](https://github.com/ekexium/dehazing-using-color-lines) and [Tomlk](https://github.com/Tomlk/Dehazing-with-Color-Lines) (also pixel-wise inference, but in scala and C++). Alternatively, consider writing a vectorized implementation. 

![Current result](images/dehaze.gif)

## How to Use 
Use a venv and install the required packages by running `pip install -r requirements.txt`.    
Execute `python dehaze.py` to dehaze `images/bricks.png`. 

## Example output
![example output](images/example.png)
![fattal result](images/fattal_result.png)

## TODO 
* Implement [automatic airlight recovery](https://www.cse.huji.ac.il/~raananf/projects/atm_light/)
