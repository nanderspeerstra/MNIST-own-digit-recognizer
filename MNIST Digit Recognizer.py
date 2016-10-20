# Author Information
__author__ = "Nander Speerstra"
__copyright__ = "Copyright (C) 2016 Nander Speerstra"
__license__ = "Public Domain"
__version__ = "1.0"


import pygame, random, os
from io import StringIO
from PIL import Image

try:
	import tkinter as tk
	from tkinter import *
	from tkinter import ttk
except ImportError:
	import Tkinter as tk
	from Tkinter import *
	import ttk

import cv2
import numpy as np

import tensorflow as tf

######################################################

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def makeImage(y_conv):
	##
	white = (255,255,255)

	##
	screen = pygame.display.set_mode((800,600))
	screen.fill(white)

	draw_on = False
	last_pos = (0, 0)
	color = (255, 128, 0)
	radius = 40

	def roundline(srf, color, start, end, radius=1):
		dx = end[0]-start[0]
		dy = end[1]-start[1]
		distance = max(abs(dx), abs(dy))
		for i in range(distance):
			x = int( start[0]+float(i)/distance*dx)
			y = int( start[1]+float(i)/distance*dy)
			pygame.draw.circle(srf, color, (x, y), radius)

	try:
		while True:
			e = pygame.event.wait()
			if e.type == pygame.QUIT:
				raise StopIteration
			if e.type == pygame.MOUSEBUTTONDOWN:
				(leftclick, middleclick, rightclick) = pygame.mouse.get_pressed()
				if leftclick:			
					color = (0,0,0)
					pygame.draw.circle(screen, color, e.pos, radius)
					draw_on = True
				elif rightclick:
					screen.fill(white)
					draw_on = False
				elif middleclick:				
					pygame.image.save(screen, "temp_img.png")
					raise StopIteration
					
			if e.type == pygame.MOUSEBUTTONUP:
				draw_on = False
			if e.type == pygame.MOUSEMOTION:
				if draw_on:
					pygame.draw.circle(screen, color, e.pos, radius)
					roundline(screen, color, e.pos, last_pos,  radius)
				last_pos = e.pos
			pygame.display.flip()

	except StopIteration:
		pass

	pygame.quit()
	makePrediction(y_conv)
	
def makePrediction(y_conv):
	
	# create an array where we can store our picture
	# Images
	images = np.zeros((1,784))
	correct_vals = np.zeros((1,10))
	
	gray = cv2.imread('temp_img.png', cv2.IMREAD_GRAYSCALE)
	#cv2.imshow('Image', gray)
	#cv2.waitKey(0)
	
	# resize the images and invert it (black background)
	gray = cv2.resize(255-gray, (28, 28))
	
	flatten = gray.flatten() / 255.0
	
	images[0] = flatten
	correct_val = np.zeros((10))
	correct_val[2] = 1 # Give incorrect score
	correct_vals[0] = correct_val

	# Print the accuracy
	#print("own  accuracy %g"%accuracy.eval(feed_dict={x: images, y_: correct_vals, keep_prob: 1.0}))
	
	# Get the prediction
	out = y_conv.eval(feed_dict={
		x: images, y_: correct_vals, keep_prob: 1.0})	
	predDigit.set('Prediction:\t{}'.format(str((np.argmax(out[0])))))
	
	# Remove temp file
	os.remove('temp_img.png')
	
#####################################

# Placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

## Now let's build the convolutional layers
# 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 3
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout (to reduce overfitting)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# The softmax layer (like the one layer softmax regression)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Start the session
with tf.Session() as sess:
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	sess.run(tf.initialize_all_variables())
	
	
	oSaver = tf.train.Saver()
		
	# Instead of training, use our already trained model!
	oSaver.restore(sess, 'mnistModelCNN.ckpt')		

	# Window
	win = tk.Tk()
	win.title('Digits recognizer')

	# first text
	textFrame1 = tk.Frame()
	mainText1 = Label(textFrame1, justify=LEFT, text='Welcome! Here, you can create your own digit!\n Can I recognize it?')
	mainText1.pack()

	textFrame2 = tk.Frame()
	mainText2 = Label(textFrame2, justify=LEFT, text='leftclick: draw, rightclick: reset, middleclick: done')
	mainText2.pack()


	imageButton = tk.Button(win, text='Create digit', command=lambda: makeImage(y_conv), height = 2, width = 10)	# Output

	predDigit = StringVar()
	predDigit.set('No prediction yet')
	predLabel = ttk.Label(win, textvariable=predDigit)
		
	# GRID
	textFrame1.grid(column=0, row=0, columnspan=3)

	textFrame2.grid(column=1, row=2, columnspan=3)

	imageButton.grid(column=1, row=4, padx=5, pady=5, columnspan=3)	

	predLabel.grid(column=1, row=6, columnspan=3)

	win.mainloop()

