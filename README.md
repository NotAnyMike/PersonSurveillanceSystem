# Person Surveillance System

Implementation of a Person Surveillance System on python, it should be able tell given two images of persons if they are the same person, the second module should be able to given some text attributes (hat, woman, etc.) retrive an images with such attributes

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

The biggest mAP comes from a classifier only using SIFT features, with binSize of 8 and step of 5
