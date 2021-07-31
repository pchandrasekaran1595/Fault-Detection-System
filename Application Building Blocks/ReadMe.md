- Common Scripts used by CLI and GUI Application
    1. DatasetTemplates.py
    2. MakeData.py
    3. Models.py
    4. Train.py
    5. utils.py

&nbsp;

- Scripts used ONLY by GUI Application
    1. gui.py

&nbsp;

- Scripts used ONLY by CLI Application
    1. cli.py      
    2. Snapshot.py
    3. RTApp.py

&nbsp;

- Scripts and their Functionalities
<pre>
1. cli.py      --> CLI Application Main
2. gui.py      --> GUI Application Main
3. main.py     --> Entry Point into the Application
4. utils.py    --> Script that contains constants and functions that are used across various scripts

5. Snapshot.py --> CLI Application Script that handles the capture of frames 
6. RTApp.py    --> CLI Application Script that handles the inference 

7. Train.py    --> Script that holds the fit() function; responsible for the entire training process of the application
8. Models.py   --> Script that holds all the models used by the application]
9. Makedata    --> Script that is responsible for the creation of the basic dataset; handles both the positive and negative classes
</pre>