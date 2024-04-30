# HSonic_OpenSource

the dataset and source code of paper "HeadSonic: Reliable Bone Conduction Earphone Authentication with Head-conducted Sounds"



Dataset：

​	Data from 45 subjects(subject ID = folder name), each subject is required to collect one session data (100 samples) at each of the four different behaviors, i.e., sitting on a chair (chair_10), rotating the head (head_10), rotating the body (body_10), and walking (walk_10).

​	Each sample has a length of 3072 (output of the SCAE model).

​	Due to GitHub's 100 MB file upload limit, the dataset was split into four parts for uploading. Before running the code, ensure that datasets for different subjects are placed within the same "mainDataset" folder.



Source code: 

​	Run "main.py" to train and test the BANN model. 

​	"main.py" automatically calls code from other files, sequentially designating each subject as a legitimate user for evaluation.

  More details could be seen within the source code.