# morelab-python
MoRe-Lab Python code for motion capture analysis

## QTM scripting
In the folder QTM, we keep Python scripts for the Qualisys Track Manager software. More information on scripting in QTM can be found here: https://github.com/qualisys/qtm-scripting

### Installation
1. Open a command prompt window as administrator
2. Go to the QTM installation folder:
   ```sh
   cd "C:\Program Files\Qualisys\Qualisys Track Manager"
   ```
3. Run this command, but replace the path with the correct path this repository:
   ```sh
   python -m pip install -r "C:\Users\MyName\Documents\Repositories\morelab-python\QTM\requirements.txt"
   ```
   This will download and install the necessary packages in your QTM Python environment.
4. In QTM Project Options->Miscellaneous->Scripting, add the script "...\morelab-python\QTM\MoreLab-Scripting.py" to the list of script files.
