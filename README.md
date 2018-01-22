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
[image6]: ./images/example_grid_1.png
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
![Eampe grid][image6]

The perspective transform function takes a source and destination set of points and outputs the bird's eye or top-down view of the image.

Since the transformed image is going to have regions that are out side of the camera vision I use a mask so that I can use this later to make sure I'm only finding obstacles, navigable surfaces, and rock in my region of interest. This is done by taking a white image and transforming is to that the white pixels represent the region of interest and the back the region outside the camera region.
This is what the mask looks like: 
![Mask][image10]

Code: 
```python
# Define a function to perform a perspective transform
# I've used the example grid image above to choose source points for the
# grid cell in front of the rover (each grid cell is 1 square meter in the sim)
# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    img_to_mask = np.ones_like(img[:,:,0])
    masked = cv2.warpPerspective(img_to_mask, M, (img.shape[1], img.shape[0]))# keep same size as input image
     
    
    return warped, masked
# Define calibration box in source (actual) and destination (desired) coordinates
# These source and destination points are defined to warp the image
# to a grid where each 10x10 pixel square represents 1 square meter
# The destination box will be 2*dst_size on each side
dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
warped, masked = perspect_transform(grid_img, source, destination)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(warped)
plt.subplot(122)
plt.imshow(masked, cmap='gray')
```

This is what the image looks like after is has been transformed
![Output: Perspective Transform image][image11]

##### Color Thresholding
Since the project is made easier for us as the obstacles and navigable terrain have separate RBG threshold, i use a simple RGB threshold value to find the ground pixels. The value I use here are the default form the lesson, and they do a good enough job to meet the criteria for project acceptance. 
```python
# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(150, 190, 180)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

threshed = color_thresh(warped)
plt.imshow(threshed, cmap='gray')
```
This is what the thresholded image looks like:
![Output: Color Threshold image][image1]

##### Coordinate Transformations
The next function is defined to turn move between the coordinate space of the world and the rover. Since the Rover's forward motion in on the x-axis, we need a set of functions that can transform the world coords to the cover coords and the rover coords to the world coords so that when you are looking between the rover and world perspectives you can align the different images, so that have the same coords. There is also a function to covert the radial cords to the rover coords.
```python
# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    typos, xpos = binary_img.nonzero()
    # Calculate pixel positions concerning the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Grab another random image
idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
warped, mask = perspect_transform(image, source, destination)
threshed = color_thresh(warped)

# Calculate pixel values in rover-centric coords and distance/angle to all pixels
xpix, ypix = rover_coords(threshed)
dist, angles = to_polar_coords(xpix, ypix)
mean_dir = np.mean(angles)

# Do some plotting
fig = plt.figure(figsize=(12,9))
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(warped)
plt.subplot(223)
plt.imshow(threshed, cmap='gray')
plt.subplot(224)
plt.plot(xpix, ypix, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
arrow_length = 100
x_arrow = arrow_length * np.cos(mean_dir)
y_arrow = arrow_length * np.sin(mean_dir)
plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)
```
Here are example output images of each function:
TopRow:
Image 1: is the original image
Image 2: is the warped image from the perspective transofrm is the mask applied.

Bottom Row: 
Image 1: thresholded image to see the navigable terrain
Image 2: The image transformed to the rover coords.

[Example output images]: ./images/coord_transform_1.png

##### Find Rocks
This function is used to find the rocks:

```python 
def find_rocks(img, thresh=(100, 200, 30)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])   
    rockpix = (img[:,:,0] > thresh[0]) \
                & (img[:,:,1] < thresh[1]) \
                & (img[:,:,2] < thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[rockpix] = 1
    # Return the binary image
    return color_select

rock_map = find_rocks(rock_img)
fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(rock_img)
plt.subplot(122)
plt.imshow(rock_map, cmap='gray')
```
Input/Output:
[Output: find rocks]: ./images/find_rocks_1.png
#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result. 

I populated the process image function as indicated in the code below
```python
# Define a function to pass stored images to
# reading rover position and yaw angle from csv file
# This function will be used by moviepy to create an output video
def process_image(img):
    
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    warped, mask = perspect_transform(img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    obs_map = np.absolute(np.float32(threshed) -1) * mask
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
   
    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    
    # 6) Convert rover-centric pixel values to world coordinates
    world_size = data.worldmap.shape[0]
    scale = 2 * dst_size
    xpos = data.xpos[data.count]
    ypos = data.ypos[data.count]
    yaw = data.yaw[data.count]

    x_world, y_world, = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)

    obsxpix, obsypix = rover_coords(obs_map)
    obs_x_world, obs_y_world, = pix_to_world(obsxpix, obsypix , xpos, ypos, yaw, world_size, scale)
    

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    data.worldmap[y_world, x_world, 2] = 255
    data.worldmap[obs_y_world, obs_x_world, 0] = 255
    nav_pix = data.worldmap[:,:,2] > 0
    
    data.worldmap[nav_pix, 0] = 0
    
        
    rock_map = find_rocks(warped, thresh=(110,110,50))
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos, ypos, yaw, world_size, scale)
        
       
        data.worldmap[rock_y_world, rock_x_world, :] = 255
        
    
    # 7) Make a mosaic image, below is some example code
    # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + Rover.worldmap.shape[0], img.shape[1]*2, 3))
        # Next you can populate regions of the image with various output
        # Here I'm putting the original image in the upper left hand corner
    output_image[0:img.shape[0], 0:img.shape[1]] = img

    # Let's create more images to add to the mosaic, first a warped image
    #warped. amsk = perspect_transform(img, source, destination)
    # Add the warped image in the upper right hand corner
    output_image[0:img.shape[0], img.shape[1]:] = warped

        # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
        # Flip map overlay so y-axis points upward and add to output_image 
    output_image[img.shape[0]:, 0:Rover.worldmap.shape[1]] = np.flipud(map_add)


        # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image
```


### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.
##### Perception Step

The perception step code is a replica of what was taught in the lesson, below are some salient points describing it:
1. setup src and dst point for transforming the image perspective.
1. transform the image and apply thresholding
1. Create a map of the obstacles using the  masked to only show obstacles in the region of interest
1. Using the Rover.vision_image set the threshold green channel to the threshold image, the red channel to the obstacles, and use the blue channel for the rocks
1.i sue the fnd_rock function to find the rocks.  

```python
# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    # Set a bottom offset to account for the fact that the bottom of the image 
    # is not the position of the rover but a bit in front of it
    # this is just a rough guess, feel free to change it!
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                  [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                  [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    # 2) Apply perspective transform
    warped, mask = perspect_transform(Rover.img, source, destination)
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed = color_thresh(warped)
    obs_map = np.absolute(np.float32(threshed) -1) * mask
    
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    Rover.vision_image[:,:,2] = threshed * 255
    Rover.vision_image[:,:,0] = obs_map * 255
    # 5) Convert map image pixel values to rover-centric coords
    xpix, ypix = rover_coords(threshed)
    
    # 6) Convert rover-centric pixel values to world coordinates
    world_size = Rover.worldmap.shape[0]
    scale = 2 * dst_size
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw

    x_world, y_world, = pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale)

    obsxpix, obsypix = rover_coords(obs_map)
    obs_x_world, obs_y_world, = pix_to_world(obsxpix, obsypix , xpos, ypos, yaw, world_size, scale)
    

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
    Rover.worldmap[y_world, x_world, 2] = 10
    Rover.worldmap[obs_y_world, obs_x_world, 0] = 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    dist, angles = to_polar_coords(xpix, ypix)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    Rover.nav_angles = angles
    
    rock_map = find_rocks(warped, thresh=(110,110,50))
    if rock_map.any():
        rock_x, rock_y = rover_coords(rock_map)
        rock_x_world, rock_y_world = pix_to_world(rock_x, rock_y, xpos, ypos, yaw, world_size, scale)
        
        rock_dist, rock_ang = to_polar_coords(rock_x, rock_y)                
        rock_idx = np.argmin(rock_dist)
        rock_xcen  = rock_x_world[rock_idx]
        rock_ycen = rock_y_world[rock_idx]

        Rover.worldmap[rock_ycen, rock_xcen, 1] = 255
        Rover.vision_image[:, :, 1] = rock_map =  255
    else: 
        Rover.vision_image[:,:,1] = 0
    
    
    return Rover
```
##### Decision Step

I did not change the Decision Step function in a big way. The only code change I made was to add check if we near a rock sample and stop the rover so we can collect the sample. See code below.

```python
# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality, but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        #Let see if we near a rock sample 
        if Rover.near_sample == 1:
            Rover.mode = 'stop'
            # Set mode to "stop" and hit the brakes!
            Rover.throttle = 0
            # Set brake to stored brake value
            Rover.brake = Rover.brake_set
            Rover.steer = 0
        else:
         # Check for Rover.mode status
         if Rover.mode == 'forward': 
             # Check the extent of navigable terrain
             if len(Rover.nav_angles) >= Rover.stop_forward:  
                 # If mode is forward, navigable terrain looks good 
                 # and velocity is below max, then throttle 
                 if Rover.vel < Rover.max_vel:
                     # Set throttle value to throttle setting
                     Rover.throttle = Rover.throttle_set
                 else: # Else coast
                     Rover.throttle = 0
                 Rover.brake = 0
                 # Set steering to average angle clipped to the range +/- 15
                 Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
             # If there's a lack of navigable terrain pixels then go to 'stop' mode
             elif len(Rover.nav_angles) < Rover.stop_forward:
                     # Set mode to "stop" and hit the brakes!
                     Rover.throttle = 0
                     # Set brake to stored brake value
                     Rover.brake = Rover.brake_set
                     Rover.steer = 0
                     Rover.mode = 'stop'
         # If we're already in "stop" mode then make different decisions
         elif Rover.mode == 'stop':
             # If we're in stop mode but still moving keep braking
             if Rover.vel > 0.2:
                 Rover.throttle = 0
                 Rover.brake = Rover.brake_set
                 Rover.steer = 0
             # If we're not moving (vel < 0.2) then do something else
             elif Rover.vel <= 0.2:
                 # Now we're stopped and we have vision data to see if there's a path forward
                 if len(Rover.nav_angles) < Rover.go_forward:
                     Rover.throttle = 0
                     # Release the brake to allow turning
                     Rover.brake = 0
                     # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                     Rover.steer = -15 # Could be more clever here about which way to turn
                 # If we're stopped but see sufficient navigable terrain in front then go!
                 if len(Rover.nav_angles) >= Rover.go_forward:
                     # Set throttle back to stored value
                     Rover.throttle = Rover.throttle_set
                     # Release the brake
                     Rover.brake = 0
                     # Set steer to mean angle
                     Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                     Rover.mode = 'forward'
    # Just to make the rover do something 
    # even if no modifications have been made to the code
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover
```
#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.  

**Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**
I ran my simulator with the following settings:
![Simulator Setting][image3]

The Rover drives and maps the areas reasonably well. It can get stuck in a few places near the bigger rocks in the middle of the navigable terrain. It also does a reasonable job of collecting the sample that is directly in the rover's path.

Things I could do to improve  this code:
1. Better thresholding and image manipulation using different color scales to better find obstacles.
1. Improve my steering to better get out of tight spots. Instead of just using a mean to define the steering angle I could use techniques to back up to a last know good path, a state machine keeping a record of the last steering angle that got me into the tricky situation in the first place and then takes a and alternative steering angle to avoid going into the same obstacle or going around in circles.
1. Develop a function that guides the rover towards the sample rocks is detects. Right now I will only collect the sample if the rock is directly in the rover's path.