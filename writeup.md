## Project: Search and Sample Return
### Writeup Template: You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---


**The goals / steps of this project are the following:**  

**Training / Calibration**  

* Download the simulator and take data in "Training Mode"
* Test out the functions in the Jupyter Notebook provided
* Add functions to detect obstacles and samples of interest (golden rocks)
* Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

**Autonomous Navigation / Mapping**

* Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.  

[//]: # (Image References)

[image1]: ./images/color_thresholding.png
[image2]: ./images/coord_transform_1.png
[image3]: ./images/screen_resolution.png 
[image4]: ./images/coord_transform_3.png 
[image5]: ./images/coord_transform_4.png 
[image6]: ./images/example_grid1.png
[image7]: ./images/example_rock1.png
[image8]: ./images/find_rocks_1.png
[image9]: ./images/find_rocks_2.png
[image10]: ./images/mask.png
[image11]: ./images/perspective_transform.png
[image12]: ./images/test_image.png


## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

You're reading it!

### Notebook Analysis
#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.
##### Perspective Transform
Here is the example grid image that I use to show the output when it is processed with my perspective transform function:
![Example grid][image6]

The perspective transform function takes a source and destination set of points and outputs the bird's eye or top-down view of the image.

Since the transformed image is going to have regions that are outside of the camera vision, I create a mask so that I can use this later to make sure I'm only finding obstacles, navigable surfaces, and rock in my region of interest. This is done by taking a white image and transforming it so that the white pixels represent the region of interest and the black area is outside the region outside the camera's vision.
This is what the mask looks like: 
![Mask][image10]

The code for the ```python perspect_transform(img, src, dst) ``` is located  in the ./RoboND-Rover-Project/code folder in the perception.py file.

This is what the image looks like after it has been transformed
![Output: Perspective Transform image][image11]

##### Obstacle Identification: Color Thresholding
Since the project is made easier for us as the obstacles and navigable terrain have separate RBG threshold, I use a simple RGB threshold value to find the ground pixels. The value I use here is the default from the lesson, and they do a good enough job to meet the criteria for project acceptance. 

The code for the ```python color_thresh(img, rgb_thresh=(150, 190, 180)): ``` is located  in the ./RoboND-Rover-Project/code folder in the perception.py file.

This is what the thresholded image looks like:
![Output: Color Threshold image][image1]

##### Coordinate Transformations
The next function is defined to move between the coordinate space of the world and the rover. Since the Rover's forward motion in on the x-axis, we need a set of functions that can transform the world coords to the cover coords and the rover coords to the world coords so that when you are looking between the rover and world perspectives you can align the different images, so that have the same coords. There is also a function to covert the radial cords to the rover coords.

The code for the ```python rover_coords(binary_img), to_polar_coords(x_pixel, y_pixel), rotate_pix(xpix, ypix, yaw),
translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale), and pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)  ``` is located  in the ./RoboND-Rover-Project/code folder in the perception.py file.

Here are example output images of each function:
TopRow:
Image 1: is the original image
Image 2: is the warped image from the perspective transform is the mask applied.

Bottom Row: 
Image 1: thresholded image to see the navigable terrain
Image 2: The image transformed to the rover coords.

![Example output images] [image2]

##### Find Rocks
A function was also defined to find rocks, again a thresholding technique was used since rock have an RGB pattern that can be identified with the values below.  

Refer to the ```python find_rocks(img, thresh=(100, 200, 30)): ``` function  located  in the ./RoboND-Rover-Project/code folder in the perception.py file.

Input/Output:
[Output: find rocks]: ./images/find_rocks_1.png


##### Summary 

The function declared in this section are critical in my perception functions of the rover as they are used to:
1. Identify obstacles using the color_threshold function and an obstacles map can be created by taking the threshold image and subtracting one from it and taking the abs value so that all the clear path show as zeros and the obstacles show as ones.
1. Again the thresholding technique can be sued to locate rocks on the map.
1. The perspective transform gives us a birds eye view of the map.
1. Masking helps us focus on the region of interest.
1. Our coordinate transformation functions help us view the images form the world or rover perspective. 


#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a world map.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

Refer to the ```python process_image(img): ``` function  located  in the ./RoboND-Rover-Project/code folder in the perception.py file.

The process image pipeline does the following things:
1. Using my source and destination point defined earlier in the Jupyter notebook I get the warped image and the mask. ```python 
   warped, mask = perspect_transform(img, source, destination) ```
1. Apply color threshold to identify navigable terrain/obstacles/rock samples ```python threshed = color_thresh(warped) ```     
1. Create an obstacle map by taking the threshold image and subtracting one from it and taking the abs value so that all the clear path show as zeros and the obstacles show as ones. ```python obs_map = np.absolute(np.float32(threshed) -1) * mask```
1. Convert map image pixel values to Rover-centric coords ```python  xpix, ypix = rover_coords(threshed)```
1. Convert rover-centric pixel values to world coordinates
```python
    world_size = data.worldmap.shape[0]
    scale = 2 * dst_size
    xpos = data.xpos[data.count]
    ypos = data.ypos[data.count]
    yaw = data.yaw[data.count]

    x_world, y_world, = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)

    obsxpix, obsypix = rover_coords(obs_map)
    obs_x_world, obs_y_world, = pix_to_world(obsxpix, obsypix , xpos, ypos, yaw, world_size, scale)
```

 1. Update Rover world map (to be displayed on right side of the screen), the world map blue channel is set to 255, and the obstacle map is shown on the map by setting the work map red channel to 255. I also find all the nav_pix where the blue chennel value are greater and 0 and then in the worldmap red channel i set the values at these array location to 0  
```python        
    data.worldmap[y_world, x_world, 2] = 255
    data.worldmap[obs_y_world, obs_x_world, 0] = 255
    nav_pix = data.worldmap[:,:,2] > 0
    
    data.worldmap[nav_pix, 0] = 0
 ```
 1. once we defined the world map with the obsicals and navigable terrain we can find the rocks on the map and update the world map so that rocks show as white at these array locations. This is done by setting all the channel to 255 at these array location in the world map array.  
     ```python data.worldmap[rock_y_world, rock_x_world, :] = 255 ```
 1. The last few lines in the function do the following:     
 1.1 Make a mosaic image
 1.1 Add the warped image in the upper right-hand corner
 1.1 Overlay world map with ground truth map
 1.1 Then putting some text over the image
    
Here is a link to the video output for this function:
[![Process Images Video output](https://github.com/nisheed75/RoboND-Rover-Project/blob/master/RoboND-Rover-Project/output/test_mapping.mp4)


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation are provided in the writeup of how and why these functions were modified as they were.
##### Perception Step
This function closely resembles the pipeline I describe above ```python process_image(img): ``` however I list some salient points here:

The perception step code is a replica of what was taught in the lesson, below are some salient points describing it:
1. Setup src and dst point for transforming the image perspective.
1. Transform the image and apply thresholding
1. Create a map of the obstacles using the  masked to only show obstacles in the region of interest
1. Using the Rover.vision_image set the threshold green channel to the threshold image, the red channel to the obstacles, and use the blue channel for the rocks
1. I use the find_rock function to find the rocks.  

Refer to the ```python def perception_step(Rover): ```  function located in the ./RoboND-Rover-Project/code folder in the perception.py file.

##### Decision Step

I did not change the decision Step function in a big way. The only code change I made was to add a check to see if we are near a rock sample and stop the rover so we can collect the sample. See code below.

Refer to the ```python decision_step(Rover): ``` function located in the ./RoboND-Rover-Project/code folder in the decision.py file.

The decision.py file includes all of the code used by the rover for decision making. It takes the sensor data from perception.py and tries to determine what actions it should make.

The main areas within the file are the forward, stop and stuck sections.

###### Forward
The forward section includes logic for calculating where the rover should drive forward. 
1. It checks the extent of navigable terrain
1.1 If mode is forward, navigable terrain looks good, and velocity is below max, then throttle otherwise coast
1.1 The average angle of the potential navigable area is used to calculate the steering angle. 
1. If there's a lack of navigable terrain pixels then go to 'stop' mode 

######Stop
The stop section is important for making the rover rotate to a new direction. As above we stop if the lack navigable terrain.
1. We make sure we have come to a complete stop if we still moving the brake is applied.
1. if we have stopped we turn -15 degrees
1. If stopped but there is navigable terrain release the brake and move forward
1. The average angle of the potential navigable area is used to calculate the steering angle. 

######Picking up Rocks
1.  If in a state where want to pickup a rock send pickup command

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**
I ran my simulator with the following settings:
![Simulator Setting][image3]

The Rover drives and maps the areas reasonably well. It can get stuck in a few places near the bigger rocks in the middle of the navigable terrain. It also does a reasonable job of collecting the sample that is directly in the rover's path.

Things I could do to improve  this code:
1. Better thresholding and image manipulation using different color scales to better find obstacles.
1. Improve my steering to better get out of tight spots. Instead of just using a mean to define the steering angle I could use techniques to back up to a last know good path, a state machine keeping a record of the last steering angle that got me into the tricky situation in the first place and then takes a and alternative steering angle to avoid going into the same obstacle or going around in circles.
1. Develop a function that guides the rover towards the sample rocks is detects. Right now I will only collect the sample if the rock is directly in the rover's path.