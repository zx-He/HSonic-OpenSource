# HSonic_OpenSource

the dataset and source code of paper "HeadSonic: Reliable Bone Conduction Earphone Authentication with Head-conducted Sounds"



Dataset：

​	Download from : https://www.kaggle.com/datasets/miracle0723/hsonic-dataset

​    Data from 45 subjects(subject ID = folder name), each subject is required to collect one session data (100 samples) at each of the four different behaviors, i.e., sitting on a chair (chair_10), rotating the head (head_10), rotating the body (body_10), and walking (walk_10).

​	Each sample has a length of 3072 (output of the SCAE model).

​	



Source code: 

​	Run "main.py" to train and test the BANN model. 

​	"main.py" automatically calls code from other files, sequentially designating each subject as a legitimate user for evaluation.

  More details could be seen within the source code.