# Person Surveillance System

Implementation of a Person Surveillance System on python, it should be able tell given two images of persons if they are the same person, the second module should be able to given some text attributes (hat, woman, etc.) retrive an images with such attributes

## Part 1 - Re-identification
The idea here is to take 2 pictures of people, extract features from both and see how they compare to each other. There are several possible ways to do that. Some discussed in the course are:
- Histogram of Gradients (HoG)
- Local binary patterns (LBP)
- Colour histogram

### Colour histogram with HSV colour space
Cell wise colour histogram is extracted from image. It is parametrised based on number of bins in the histogram and the cell size. Let's play with both!
However, in Lecture 4.2.1, Tim suggests it's best to use colour histogram + HoG + LBP. Let's try that and maybe quote his paper?

HSV Colour space seems to work way better in this usage.

| Window size | 20 bins | 40 bins | 60 bins | 100 bins | 120 bins | 140 bins|
| -- |-- |-- |-- |-- |-- |-- |
| 16 16 | 30.37 | 25.96 | 27.47 | 24.61 | 25.66 | 24.13 | 22.49 | 
| 32 16 | 29.06 | 30.29 | 31.29 | 31.50 | 27.48 | 31.48 | 29.33 |
| 32 32 | 24.25 | 25.19 | 26.47 | 27.01 | 28.93 | 29.37 | 27.01 |
| 64 32 | 23.48 | 27.29 | 29.43 | 28.64 | 31.54 | 29.86 | 30.43 |

### Colour histogram with RGB colour space

In addition to the HSV colour histogram, now I'll also stack an RGB one

| nbin | with total | without total |
| -- | -- | -- |
| 5  | 33.59  | 32.20 | 
| 10 | 31.96  | 32.70 |
| 15 | 31.92  | 32.76 |
| 20 | 31.90  | 32.97 |
| 30 | 31.86  | 33.14 |
| 40 | 31.39  | 33.14 |
| 60 | 32.25  | 33.67 |
| 80 | 31.46  | 33.07 |
| 100 | 31.76 | 32.49 |

### Histogram of Gradients (HoG)
default params:
- win_size = [16 16];
- nbins = 24;
- block_size = [4 4];

Acc: 11.07%

| nbins | mAP |
| -- | -- |
| 10 | 38.45% |
| 15 | 40.11% |
| 20 | 40.06% |
| 30 | 42.40% |
| 40 | 42.68% |
| 50 | 38.47% |

### Local binary patterns (LBP)
default params:
- win_size = [16 16];
- n_neighbour = 8;

Acc: 13.44%

Can't really change the window size, so let's play with the neighbours

| neighbour | mAP |
| -- | -- |
| 4  | 35.42 |
| 5  | 32.78 |
| 6  | 34 83 |
| 7  | 34.10 |
| 8  | 37.41 |
| 9  | 36.39 |
| 10 | 32.46 |
| 11 | 33.59 |
| 12 | 33.12 |


### Tuning the SVM
With all of the 3 features from above concatenated

| outliers | mAP |
| -- | -- |
| 0.00 | 27.89 |
| 0.05 | 27.19 |
| 0.1  | 27.51 |
| 0.15 | 25.67 |
| 0.09 | 28.33 |

| outliers | mAP |
| -- | -- |
| 0.1 | 27.64 |
| 0.2 | 27.90 |
| 0.3  | 27.75 |
| 0.25 | 28.91 |
| 0.22 | 28.98 |

Tuned SVM will be refereed to as SVM+.

### Correlogram
Implementation used from the official MATLAB package. Increases performance on average by approx 2%.

### SIFT
Explored but produced bad results. Doesn't really make sesnse to use SIFT here as all our our images have similar orientation and scale. **Not used.**

### MSCR
Used by itself and produced OK results (~26%) but when combines with everything else it deteriorates. **Not used.**



### Putting it all together
Results:
| Feature extractor | Comparator | Avg. precision |
| -- | -- | -- |
| HoG | SVM | 11.07 |
| HoG + Colour histogram | SVM | 13.12 |
| HoG + Colour histogram + LBP | SVM | 13.14 |
| Colour histogram | SVM | 11.00 |
| Colour Histogram + HoG + LBP | SVM+ | 28.99 |
| Correlogram + Colour histogram + LBP | SVM-  | 31.07 |
| Correlogram + Colour histogram + HOG + LBP | SVM-one | 35.49 |


### Final parameters
```
% parameters for correlogram
use_correlogram = true;
correlogram_window = [128 64];
correlogram_fun = @(block_struct) colorAutoCorrelogram(block_struct);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for colour histogram
use_colour = false;
use_colour_hsv = true;
colour_nbin = 4;
colour_win_size = [16 16];
fun = @(block_struct) histcounts(block_struct.data,colour_nbin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for HoG
use_hog = true;
hog_win_size = [16 16];
hog_nbins = 4;
hog_block_size = [2 2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   %

% parameters for LGP
use_lgp = true;
lbp_win_size = [16 16];
lbp_n_neighbour = 8;
lgp_radius = 1;
is_upright = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```


## Things to try

** Colors are very useful **

## Results (Module 2)

these are the results

| Configuration | Using CrossVal | mAP |
|:-------------:|:--------------:|:---:|
| Base System   | No | 33% |
| Hog Attributes with 16 wind and 24 bins and block of 4 | No | 37% |
| HoG Attributes | Yes | 35% |
| BoW with HOG attributes | No | 29% |
| BoW with SIFT attributes | No | 32% |
| BoW with SIFT attributes | Yes| 32% |
| SIFT attributes | Yes | 27% |
| SIFT attributes | No | 36% |
| Colors hist with norm | No | 32% |
| Window 16 of hist colors | No | 38% |
| Window 16 of hist colors with sum | No | 39% |
| Window 16 of hist colors with sum and total | No | 40% |
| LBP | | 36% |
