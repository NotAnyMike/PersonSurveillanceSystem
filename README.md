# Person Surveillance System

Implementation of a Person Surveillance System on python, it should be able tell given two images of persons if they are the same person, the second module should be able to given some text attributes (hat, woman, etc.) retrive an images with such attributes

## Things to try

* Implement Histogram of Colors

## Results (Module 2)

these are the results

| Configuration | Using CrossVal | mAP |
|:-------------:|:--------------:|:---:|
| Base System   | No | 35% |
| HoG Attributes | Yes | 35% |
| BoW with HOG attributes | No | 29% |
| BoW with SIFT attributes | No | 32% |
| BoW with SIFT attributes | Yes| 32% |
| SIFT attributes | Yes | 27% |
| SIFT attributes | No | 36% |
| LBP attributes | No | 37% |

The biggest mAP comes from a classifier using only LBP features with cell size of 8x8 and the second best comes from a model only using SIFT features, with binSize of 8 and step of 5 as well as 
