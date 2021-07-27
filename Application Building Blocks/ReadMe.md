- Common Scripts used by CLI and GUI Application
    1. DatasetTemplates.py
    2. MakeData.py
    3. Models.py
    4. Train.py
    5. utils.py
    
- Scripts used ONLY by GUI Application
    1. gui.py

- Scripts used ONLY by CLI Application
    1. cli.py      
    2. Snapshot.py
    3. RTApp.py

- Scripts and their Functionalities
    1. cli.py   --> CLI Application
    2. gui.py   --> GUI Application
    3. main.py  --> Entry Point into the Application
    4. utils.py --> Script that contains constants and fucntions that are used across various scripts

    5. Snapshot.py --> CLI Application Script that handles the capture of frames 
    6. RTApp.py    --> CLI Application Script that handles the inference 
 
    7. Train.py  --> Script that holds the fit() function; also responsible for splitting data import Train/Valid Splits
    8. Models.py --> Script that holds all the models used by the application