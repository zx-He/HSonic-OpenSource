# HSonic_OpenSource

the dataset and source code of paper "HeadSonic: Reliable Bone Conduction Earphone Authentication with Head-conducted Sounds"



Dataset：

​    Data from 45 subjects, each subject is required to collect one session data (100 samples) at each of the four different behaviors, i.e., sitting on a chair (chair.pth), rotating the head (head.pth), rotating the body (body.pth), and walking (walk.pth).

​	Each sample has a length of 3072 (output of the SCAE model).

​	

Source code: 

​	Run "BANN_transfer.py" to for the BANN model transfer learning. 

​	"BANN_transfer.py" automatically calls code from other files, sequentially designating each subject as a legitimate user for evaluation.

  More details could be seen within the source code.

