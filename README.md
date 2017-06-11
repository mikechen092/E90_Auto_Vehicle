# E90-Autonomous-Vehicle
by Chris Grasberger & Mike Chen

The goal of this project was to create an autonomous vehicle that would be capable of traveling the paths of campus. The framework we started with was simply the chassis and the motors. A motor controller, self tuning PID controller, deep cycle solar batteries, a laptop, a small web camera, and a video game controller were acquired to create our autonomous vehicle. Using Robot Operating System (ROS), we implemented a vision node as well as a controller node. The vision node utilized a support vector machine to classify on every image taken from the camera and classified each pixel as path or not path. This information was then sent to the controller node which processed the data from the vision node to calculate individual wheel speeds. Unfortunately we were not able to fully accomplish our goal of creating a fully autonomous vehicle, but we were able to create a solid framework capable of path detection and following.  

The color finding folder contains code we initially used to try and find objects
in an image of a certain color. This includes gray finding as well as blue finding, although these two are essentially using the same algorithm.

The ros folder contains the code we used that has been ros-ified to use within
the ros framework. It contains the controller node, the vision node, as well as
the trained SVM within picklejar.pkl. The two vision nodes contain the color
finding version as well as the machine learning version.

The ml_data contains files to train the support vector machine with. It has
a test set as well as a target set. This data set is by no means comprehensive.
There needs to be more variety added to be fully capable of path detection.

The results folder contains images of the results of our vision process

Con.py is a python script taken from Noah Weinthal and Neil MacFarland E90 project
that allows python communication with a controller. It is imported into other
scripts to be used.

Game.py is a python script to connect to the appropriate port to manually control
the vehicle with a game controller. You can manually control each wheel speed from a gaming controller.

Cvk2.py is a script written by Matt Zucker. It allows additional fuctionality
when using findContours which aids in the process.
