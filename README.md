As i am unaware of web frameworks i have used advanced machine learning techniques to detect accident in video.
SUMMARY:
Advanced machine learning techniques are employed to compare video footage containing accidents to video free of crashes. Minute differences can be found using a recurrent neural network from a labeled video set. These differences can then be used to find accidents in videos.
PROCESS:
Each video is broken up into its individual frames to be analyzed separately. Each of these images is a two-dimensional array of pixels where each pixel has information about the red, green, and blue  color levels. To reduce the dimensionality at the individual image level, I convert the 3-D RGB color arrays to grayscale. Additionally, to make the computations more tractable on a CPU, I downsample each image by a factor of 5 - in effect, averaging every five pixels to reduce the size of each image to a 2-D array .

Additional processing was explored for this project, such as median-subtracting out the background of the images and featurizing each frame in the image. 
I have used HRNN algorithm which is used to tackle the complex problem of classifying the video footage.
