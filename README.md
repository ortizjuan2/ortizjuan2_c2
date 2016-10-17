# Udacity Challenge 2: Using Deep Learning to Predict Steering Angles 
### by Juan C. Ortiz

For more details see [Challenge 2](https://medium.com/udacity/challenge-2-using-deep-learning-to-predict-steering-angles-f42004a36ff3#.32gnncto4).

## Extracting Camera images and angles

Driving datasets are available in rosbag format and images can be compressed.
To extract data use gen_imgs.py in scripts directory, see help for command line
parameters. Please be patiend, with large rosbag files it can take longer.

example:
Take into account the compressed parameter is related to the image comming in the datased
not to the output itself.

python ./scripts/gen_imgs.py --camera dataset.bag -- steering dataset.bag --compressed yes


To see extracted images run:
python ./scripts/get_imgs.py --imgs "generated images file" --angles "generated angles file"


## Model definition and training

Still working on that...

