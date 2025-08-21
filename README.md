Summary
An Audio File Management Specialist is needed to organize, rename, and archive MP3 files from multiple sources, leveraging AI and software tools to automate the process.


General Information
The project involves managing daily generated MP3 files from Google Drive and local Mac storage. The goal is to filter, sort, and store relevant files on a network storage device, with accurate and descriptive file titles that include speaker identification when possible. Knowledge of Audio Hijack is beneficial.


Tasks and Deliverables
- Aggregate MP3 files from Google Drive and local Mac storage (Audio Hijack folder).
- Only process audio files that are 10 minutes or more
- support for .mp3 and .wav files
- transcribe audio files to have the context of the files
- save the transcript
- The folder structure should be as follows: 
	- speakers (sample voices of speakers, will be used to train the algorithm that will figure out which speaker it is) 
		- [NAME OF THE SPEAKER].mp3
      - talks 
        - [YEAR]
          - [SPEAKER NAME]
          	- [SPEAKER NAME] - [3 WORD DESCRIPTION].mp3
          	- [SPEAKER NAME] - [3 WORD DESCRIPTION] (Transcript).txt
	- raw talks
- Delete junk files every month 
- Record the last date of a successful run, and just get the files that are from that date until the current date
- Rename MP3 files to include relevant data, such as speaker identification.
- Automate the process to regularly update and archive files to the network storage device.
- The whole workflow needs to be optimized to run on Apple Silicon (MLX)
