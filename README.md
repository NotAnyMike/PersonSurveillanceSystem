# Person Surveillance System

Implementation of a Person Surveillance System on python, it should be able tell given two images of persons if they are the same person, the second module should be able to given some text attributes (hat, woman, etc.) retrive an images with such attributes

## Results (Module 2)

| Configuration | Using CrossVal | mAP |
|:-------------:|:--------------:|:---:|
| Base System   | No | 35% |
| HoG Attributes | Yes | 35% |
| BoW with SIFT Attributes | No | 30% |
| SIFT attributes | Yes | 27% |
| SIFT attributes | No | 27% |