Implementation assignment 3
By: Evan Medinger, Rajat Kulkarni, Logan Saso

To run the program, call 'py main.py' in the terminal when in
the src directory. By calling this command, will run the basic
functions of the program.

There are a few setting that can be manipulated in this program
for ease of the user. In the __name__ of the main.py, they
are labelled as so:

	#####ADDED VARIABLES####
	plotValues = False	# set to true to generate images
	setting = 0 		# 0 = depth testing, 1 = tree testing, 2 = feature testing
	########################

When plotValues is set to true, it will render a graph respective
to the number entered inside. A 0 value will develop a graph that
affects the change in max_depth, a 1 value will affect n_tree of
the forest generated, and a 2 will affect max_features that are
available to tree to generate from. These functions are originally
turned off due to how time consuming these functions are.