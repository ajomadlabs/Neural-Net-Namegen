'''--------------------------------------------------------------------------------------------------
	Importing The Important modules
--------------------------------------------------------------------------------------------------'''

from __future__ import absolute_import, division, print_function					#Importing the modules such that they work the same in both versions of python
import os												#Importing the os module
from six import moves											#Importing the six module
import ssl												#Importing the ssl module
import tflearn												#Importing the tflearn module
from tflearn.data_utils import *									#Import all the data utils in tflearn

'''-----------------------------------------------------------------------------------------------'''

'''--------------------------------------------------------------------------------------------------
	Getting The Dataset For Generating The New Data
--------------------------------------------------------------------------------------------------'''

path = "US_Cities.txt"											#Setting the path variable to the dataset
if not os.path.isfile(path):										#CHecking if the path exists
    contxt = ssl._create_unverified_context()								#Creating a variable and assigning ssl 
    moves.urllib.request.urlretrieve("https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/US_Cities.txt", path, context=contxt) #Pulling the dataset from the corresponding url

'''-----------------------------------------------------------------------------------------------'''

'''--------------------------------------------------------------------------------------------------
	Setting The MaxLength Of The New City Names Generated
--------------------------------------------------------------------------------------------------'''
		
maxlength = 20												#Setting the maximum length of a newly generated city name

'''-----------------------------------------------------------------------------------------------'''

'''--------------------------------------------------------------------------------------------------
	Vectorising The Dataset That Is Loaded
--------------------------------------------------------------------------------------------------'''

X, Y, char_dic = textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlength, redun_step=3)		#Creating a vector from the dataset which finds the commonalities among all the 													dataset thats imported

'''------------------------------------------------------------------------------------------------'''

'''---------------------------------------------------------------------------------------------------
	Creating A Long Short Term Memory Neural Network
---------------------------------------------------------------------------------------------------'''

lstm = tflearn.input_data(shape=[None, maxlength, len(char_dic)])					#Creating the input layer 
lstm = tflearn.lstm(lstm, 512, return_seq=True)								#Creating the second layer consisting of 512 nodes
lstm = tflearn.dropout(lstm, 0.5)									#Creating a dropout. A dropout will help to off some nodes to avoide overfitting
lstm = tflearn.lstm(lstm, 512)										#Creating the third layer consisting of 512 nodes
lstm = tflearn.dropout(lstm, 0.5)									#Creating a drop out. This also helps to generalise the system
lstm = tflearn.fully_connected(lstm, len(char_dic), activation='softmax')				#Creating the output layer using softmax
lstm = tflearn.regression(lstm, optimizer='adam', loss='categorical_crossentropy',			#Creating the regression for the nodes created
                     	  learning_rate=0.001)

'''-------------------------------------------------------------------------------------------------'''

'''----------------------------------------------------------------------------------------------------
	Creating The Learning Process Of The Neural Network
----------------------------------------------------------------------------------------------------'''

city_gen = tflearn.SequenceGenerator(lstm, dictionary=char_dic,						#Generating the new city names by using a sequence generator function of tensor flow
                              seq_maxlen=maxlength,
                              clip_gradients=5.0,
                              checkpoint_path='model_us_cities')

'''-------------------------------------------------------------------------------------------------'''

'''----------------------------------------------------------------------------------------------------
	Training The Neural Network For City Generation
----------------------------------------------------------------------------------------------------'''
	
for i in range(40):											#A loop which runs the training for 40 times
    seed = random_sequence_from_textfile(path, maxlength)						#A random city name is used from the file and assigned to seed variable
    city_gen.fit(X, Y, validation_set=0.1, batch_size=128,						#Training is done using the fit method
          n_epoch=1, run_id='us_cities')
    print("-- TESTING...")										#Printing 'TEST'
    print("-- Test with temperature of 1.2 --")								#Printing 'Tempurature'
    print(m.generate(30, temperature=1.2, seq_seed=seed))						#Tempurature set to 1.2
    print("-- Test with temperature of 1.0 --")								#Printing 'Tempurature'
    print(m.generate(30, temperature=1.0, seq_seed=seed))						#Tempurature set to 1.2
    print("-- Test with temperature of 0.5 --")								#Printing 'Tempurature'
    print(m.generate(30, temperature=0.5, seq_seed=seed))						#Tempurature set to 1.2

'''-------------------------------------------------------------------------------------------------'''
