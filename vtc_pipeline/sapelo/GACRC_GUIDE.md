# Running VTC on GACRC (Sapelo2) — step-by-step

VTC labels who is speaking when (adult female, adult male, key child, other children). It needs a Linux computer with a GPU, so it runs on UGA's Sapelo2 cluster. The result is a file called `rttm.csv`, which you then open in the conversational turns app on your own computer.

**Where to type each command:**
- 💻 **Your PC** = PowerShell on your Windows computer
- 🖥️ **Sapelo2** = the PowerShell window *after* you've logged in to Sapelo2 (the prompt will start with your MyID, e.g. `MyID@ss-sub1 ~$`)

Everywhere below, replace **`MyID`** with your UGA MyID.

---

## Before you start

- **A Sapelo2 user account.** Being in a lab group isn't quite enough on its own: each person needs their own Sapelo2 account linked to the group. If you've never logged in, request an account for yourself through GACRC: https://wiki.gacrc.uga.edu/wiki/Getting_Started
- **UGA VPN** if you're off campus (the "Remote Access VPN"). On campus you don't need it.
- **Duo** on your phone, for the login prompt.

---

## Part 1 — One-time setup (about 15–20 minutes)

You only do this once.

### 1.1 Log in

💻 **Your PC:**
```
ssh MyID@sapelo2.gacrc.uga.edu
```
Enter your UGA password (nothing appears as you type; that's normal), then approve the Duo prompt.

### 1.2 Install VTC

🖥️ **Sapelo2:** copy and paste this whole block, then press Enter:

```bash
mkdir -p /scratch/$USER/vtc/{code,incoming,logs}
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd /scratch/$USER/vtc/code
git clone https://github.com/LAAC-LSCP/VTC.git
cd VTC
pip install --user "huggingface_hub[cli]"
huggingface-cli download coml/VTC-2 --local-dir ./VTC-2
uv sync
cd /scratch/$USER/vtc
wget -O run_vtc_lab.sbatch https://raw.githubusercontent.com/lestep1225118/preschooler-distress-pipeline/master/vtc_pipeline/sapelo/run_vtc_lab.sbatch
echo "SETUP DONE"
```

It will print a lot of text while it downloads. When you see **SETUP DONE**, it's finished.

Type `exit` to log out.

---

## Part 2 — Each time you have new recordings

### 2.1 Upload your audio

💻 **Your PC:** open PowerShell in the folder that contains your recordings (right-click → Open in Terminal), then run:

```
scp *.wav MyID@sapelo2.gacrc.uga.edu:/scratch/MyID/vtc/incoming/
```

(Use `*.mp3` instead if your files are MP3. To upload one file, use its name instead of `*.wav`.) It will ask for your password and Duo again. Large files take a few minutes.

You don't need to convert the audio first. The job converts it to the format VTC needs automatically.

### 2.2 Start the job

💻 **Your PC:** log in:
```
ssh MyID@sapelo2.gacrc.uga.edu
```

🖥️ **Sapelo2:**
```bash
cd /scratch/$USER/vtc
sbatch run_vtc_lab.sbatch
```

It should reply `Submitted batch job 12345678`.

### 2.3 Check on it

🖥️ **Sapelo2:**
```bash
squeue -u $USER
```

- `PD` = waiting in line for a GPU (can take a few minutes to a few hours if Sapelo2 is busy)
- `R` = running
- Nothing listed = finished

You can log out (`exit`) while it runs and check back later.

### 2.4 Check that it worked

🖥️ **Sapelo2:**
```bash
cd /scratch/$USER/vtc
tail -3 logs/vtc_*.out
head out/rttm.csv
```

If it worked, the last line says **Done**, and `rttm.csv` shows rows with a file name, a start time, a duration, and a speaker label (FEM, MAL, KCHI, or OCH).

### 2.5 Download the results

💻 **Your PC** (in the folder where you want the file):
```
scp MyID@sapelo2.gacrc.uga.edu:/scratch/MyID/vtc/out/rttm.csv .
```

(The `.` at the end means "this folder".)

### 2.6 Get the turn counts

Open the conversational turns app as usual (`py -3.12 -m vtc_pipeline.gui_app`), upload `rttm.csv`, and click **Compute turns**. Each recording appears as its own row, named after its file.

### 2.7 Clear out the old audio

Before the next batch, remove the previous recordings so they aren't processed again.

🖥️ **Sapelo2:**
```bash
rm /scratch/$USER/vtc/incoming/*
```

---

## Good to know

- **Scratch is temporary storage.** Files there are deleted automatically after a period of not being used, so download `rttm.csv` soon after each run. Keep your original recordings on your own computer or lab storage.
- **One job handles many files.** Put all the recordings in `incoming/` and they're processed together into one `rttm.csv`.
- **Each new run replaces `out/rttm.csv`**, so download it before starting another job.

## If something goes wrong

🖥️ **Sapelo2:**
```bash
cd /scratch/$USER/vtc
tail -20 logs/vtc_*.err
```

Send a screenshot of that output.

| Problem | Likely fix |
|---|---|
| `Permission denied` when logging in | Check your MyID/password; off campus, connect to the UGA VPN first |
| `Connection timed out` | Not on the VPN |
| `No audio files found` in the log | The upload in step 2.1 didn't land in `incoming/`; check `ls /scratch/$USER/vtc/incoming` |
| `sbatch: error: ... DOS line breaks` | Run `sed -i 's/\r$//' run_vtc_lab.sbatch`, then submit again |
| `huggingface-cli: command not found` during setup | Run `export PATH="$HOME/.local/bin:$PATH"` and rerun the setup block |
