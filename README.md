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
3. Run this command, but replace the path with the correct path to this repository:
   ```sh
   python -m pip install -r "C:\Users\MyName\Documents\Repositories\morelab-python\QTM\requirements.txt"
   ```
   This will download and install the necessary packages in your QTM Python environment.
4. In QTM Project Options->Miscellaneous->Scripting, add the script "...\morelab-python\QTM\MoreLab-Scripting.py" to the list of script files.



### Auto-labelling

We have implemented an algorithm that automatically labels markers by comparing **distance distributions** between marker pairs.  
This requires a **reference distribution** generated from a well-labelled, representative recording.  

> 🔑 The algorithm works best if the reference recording comes from **the same person performing the same task** as in the recording being labelled.

---

#### Step-by-step guide

1. **Generate a reference distribution**  
   From a well-labelled recording, select **Generate reference distribution** in the menu.  
   This will save the distribution as a `.npz` file.  

2. **Run auto-labelling on a new recording**  
   - Open the recording you want to label.  
   - Select **Auto label everything**.  
   - When prompted, load the `.npz` reference distribution.
   - ℹ️ The algorithm will first evaluate existing labels (removing poor ones) and then attempt to label previously unlabelled trajectories.  

3. **Handle short parts**  
   - Segments shorter than **20 frames** will *not* be labelled with **Auto label everything**.  
   - To label short segments, select them manually and run **Auto label selected trajectories (only if no overlap)**.  
   - ℹ️ This method is more conservative because it does not remove existing labels.  
   - 🔑 Recommended: Use this for all parts longer than **5 frames**.  

4. **Refill gaps**  
   Auto-labelling will undo any gap filling. To restore gaps:  
   - Go to **Capture → Reprocess**  
   - Enable **Gap-fill the gaps**, leaving all other options unchecked.
   - Reprocess.

---

#### 💡 Tips

- **Auto label everything** is eqivalent to first running **Auto label labeled**, then **Auto label unlabeled**.  
- For live feedback, run the algorithm from the **Terminal**:  
  - `gui_auto_label_everything()`  
  - `gui_auto_label_labelled()`  
  - `gui_auto_label_unlabelled()`  
  - `gui_auto_label_selected_trajectories()`  
- When running from the Terminal, you can also specify the reference file directly, e.g.:  
  ```sh
  gui_auto_label_selected_trajectories("My distribution.npz")

## Project database (experimental!)

### Installation

Creates and activates a conda environment named "morelab":
```sh
conda env create -f project/environment.yml
conda activate morelab
```

A project database can be created with the `new_database.sh` script (run it on the MoreLab server).

Create a `.env` file in your working directory with the following information (needed to connect to the database):
```sh
DB_HOST=lfs1370.srv.lu.se
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=
SSH_HOST=lfs1370.srv.lu.se
SSH_PORT=22
SSH_USER=med-pha
SSH_KEY=~/.ssh/id_rsa
SSH_PASSPHRASE=
```

### Usage

This will read the QTM project structure from `Settings.paf`, scan the data directory for qtm files and register them in the database:
```sh
python -m project.register_qtm_files /MyQtmProject/Settings.paf /MyQtmProject/Data --dry-run
```
Remove `--dry-run` if you want to make changes to the database.