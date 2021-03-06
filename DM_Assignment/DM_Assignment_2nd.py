
# coding: utf-8

# In[1]:

import array
import numpy as np
import math
from sklearn.linear_model import LinearRegression


# In[2]:

Count_Of_Messages = array.array('i',(0 for i in range(0, 1440)))
Missing_Minutes = array.array('b', (False for i in range(0, 1440)))

# Since the messages should be output in time sorted order, maintaining a 2D array
Message_String = []
for MinCount in range(1440):
    Message_String.append([])
    for Second in range(60):
        Message_String[MinCount].append([])


# In[7]:

# Parse the text file and store the count of messages in each minute and store them
with open('data/input_file.txt') as f:
    FileContent = f.readlines()
idx = 0
LastIdx = 0 # Store the last hour/minute combination present in the input just to stop processing after this point
for EachLine in FileContent:
    idx += 1
    if(idx != 1): # Ignoring the 1st line since it is empty
        SplitLine = EachLine.split()

        try:
            TimeSplit = SplitLine[3].split(':')
            TimingInMins = int(TimeSplit[0]) * 60 + int(TimeSplit[1])
            Count_Of_Messages[TimingInMins] += 1 # Increment the message count
            
            # Add all the messages in this hour/minute combo to the list at the position of the 'seconds' index
            Message_String[TimingInMins][int(TimeSplit[2])].append(EachLine) 

            if(LastIdx < TimingInMins):
                LastIdx = TimingInMins

        except:
            print("Error in getting the count information", idx)


# In[8]:

# Loop through the Message Counts and Compute the Message Count for the missing minutes
LoopIdx = -1
for EachMsgCount in Count_Of_Messages:
    LoopIdx += 1    
    if (EachMsgCount == 0):
        Missing_Minutes[LoopIdx] = True # Set this array index as the missing minute        
        InputDataSet = []
        OutputDataSet = []
        
        # Message count 2 mins before
        InputDataSet.append(LoopIdx - 2)
        OutputDataSet.append(Count_Of_Messages[LoopIdx - 2])
        
        # Message count 1 min before
        InputDataSet.append(LoopIdx - 1)
        OutputDataSet.append(Count_Of_Messages[LoopIdx - 1])
        
        # Get the count of messages received in the next 2 minutes
        NextMinIdx = 0
        IdxToTraverse = LoopIdx
        while(NextMinIdx != 2) :
            IdxToTraverse += 1
            if(Count_Of_Messages[IdxToTraverse] != 0):
                NextMinIdx += 1
                InputDataSet.append(IdxToTraverse)
                OutputDataSet.append(Count_Of_Messages[IdxToTraverse])
        
        InputDataSet = np.array(InputDataSet,np.integer).reshape(len(InputDataSet), 1)
        OutputDataSet = np.array(OutputDataSet,np.integer).reshape(len(InputDataSet), 1)
        
        Regr_Model = LinearRegression()
        
        Regr_Model.fit(InputDataSet, OutputDataSet)
        
        Predict_y = Regr_Model.predict(LoopIdx)
        Count_Of_Messages[LoopIdx] = Predict_y # Set the predicted data as the message count of the missing minute
                
    if(LoopIdx == LastIdx): # Break the loop if last index is reached
        break;


# In[10]:

# Create the output file with the updated text
EachMinIdx = 0
with open('data/output_file.txt', 'w') as outfile:
    while(EachMinIdx < 1440):
        if(Missing_Minutes[EachMinIdx] == True): # For missing minute, print the given statement based on the predicted count
            MessageCount = 0
            while(MessageCount < Count_Of_Messages[EachMinIdx]) :
                outfile.write("Mon Feb 29 "+ str(math.floor(EachMinIdx/60)) + ":"+ str(EachMinIdx % 60) + ":00 missing text here" + "\n")
                MessageCount += 1
        else: # For present minute, print the given messages in sorted order
            for EachMinMessage in Message_String[EachMinIdx]: 
                for EachSecMessage in EachMinMessage:
                    outfile.write(EachSecMessage)
        EachMinIdx += 1

print("Completed. Output file stored in the same path as the file.")


# In[ ]:



