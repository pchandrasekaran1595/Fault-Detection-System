### Repo consisiting of various approaches towards implementing a real-time Fault Detection System.

- Fixes: Add Exception Handling

- To build an exe using pyinstaller, go into the directory containing `main.py` and run the command,

`pyinstaller main.py --onefile --clean --distpath "./" --name Application --hidden-import=skimage.filters.rank.core_cy_3d --log-level=DEBUG`
