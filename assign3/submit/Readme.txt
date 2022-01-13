1. Activate python virtual environment in the same directory where the code is.

	source project_name/bin/activate

2. To be able to run this code on the “babylon01” server under the virtual environment, please have these packages installed:
a. Numpy
b. Pandas
c. Matplotlib

One of the way to install packages is to use pip install

ex) 	pip install numpy
	pip install pandas
	pip install matplotlib

If you have problem about pip, please try pip install --upgrade pip

3. Next, to run the program, please use the following command:
	
	python test.py

4. Probably, matplot graphs won't be able to show on terminal console. Please check png files in the Zip file or PDF report file.

5. I seperated L1 regularization file and L2 regularization file. the difference is just regularization part.

6. competition.py and 2competiton.py are used for Kaggle competition training. The number of features is much higher than L1 and L2 regularization files. So, it takes a long time to finish it because this code needs to compute a large NumPy array size.

7. Although the instruction said one file for each part, I made two files. You can choose one of competition.py or 2competiton.py. But, the difference is just weight size and one-hot encoding.
