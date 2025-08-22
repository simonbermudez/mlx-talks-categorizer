# Setting Up MLX Talks Categorizer as a Cronjob on macOS

This guide will help you set up the MLX Talks Categorizer to run automatically as a cronjob on macOS.

## Quick Setup (Recommended)

**Use the automated setup script for the easiest installation:**

```bash
./setup_cronjob.sh
```

This script will automatically:
- Detect your project directory and create absolute paths
- Update config.json with proper paths
- Create and test the wrapper script
- Set up your preferred schedule (including daily at 9 PM option)
- Choose between crontab or launchd setup methods
- Generate monitoring scripts

**Skip to [Prerequisites](#prerequisites) if you prefer manual setup.**

## Prerequisites

1. **Full Disk Access**: Grant Terminal (or your terminal app) Full Disk Access in System Preferences > Security & Privacy > Privacy > Full Disk Access. This is required for accessing files in folders like Google Drive.

2. **Working Installation**: Ensure the MLX Talks Categorizer is working correctly when run manually.

## Step 1: Create a Wrapper Script

Create a wrapper script that activates the virtual environment and runs the audio manager:

```bash
# Create the wrapper script
cat > /Users/$(whoami)/Dev/mlx-talks-categorizer/run_audio_manager.sh << 'EOF'
#!/bin/bash

# Set the working directory
cd /Users/$(whoami)/Dev/mlx-talks-categorizer

# Activate the virtual environment
source venv/bin/activate

# Set environment variables if needed
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

# Run the audio manager with logging
python3 main.py >> cronjob.log 2>&1

# Log completion
echo "$(date): Cronjob completed" >> cronjob.log
EOF

# Make the script executable
chmod +x /Users/$(whoami)/Dev/mlx-talks-categorizer/run_audio_manager.sh
```

## Step 2: Test the Wrapper Script

Before setting up the cronjob, test the wrapper script manually:

```bash
# Navigate to the project directory
cd /Users/$(whoami)/Dev/mlx-talks-categorizer

# Run the wrapper script
./run_audio_manager.sh

# Check the log file
tail -n 20 cronjob.log
```

## Step 3: Set Up the Cronjob

### Option A: Using crontab (Recommended)

1. **Open crontab for editing**:
   ```bash
   crontab -e
   ```

2. **Add one of these cronjob entries** (choose based on your needs):

   **Daily at 2 AM**:
   ```
   0 2 * * * /Users/$(whoami)/Dev/mlx-talks-categorizer/run_audio_manager.sh
   ```

   **Daily at 9 PM**:
   ```
   0 21 * * * /Users/$(whoami)/Dev/mlx-talks-categorizer/run_audio_manager.sh
   ```

   **Every 6 hours**:
   ```
   0 */6 * * * /Users/$(whoami)/Dev/mlx-talks-categorizer/run_audio_manager.sh
   ```

   **Weekly on Sunday at 3 AM**:
   ```
   0 3 * * 0 /Users/$(whoami)/Dev/mlx-talks-categorizer/run_audio_manager.sh
   ```

   **Every hour** (for testing):
   ```
   0 * * * * /Users/$(whoami)/Dev/mlx-talks-categorizer/run_audio_manager.sh
   ```

3. **Save and exit** (in vim: press `Esc`, type `:wq`, press `Enter`)

### Option B: Using launchd (macOS Native)

Create a Launch Agent plist file:

```bash
# Create the plist file
cat > ~/Library/LaunchAgents/com.mlx.talks.categorizer.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.mlx.talks.categorizer</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/$(whoami)/Dev/mlx-talks-categorizer/run_audio_manager.sh</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/Users/$(whoami)/Dev/mlx-talks-categorizer</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/Users/$(whoami)/Dev/mlx-talks-categorizer/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/$(whoami)/Dev/mlx-talks-categorizer/launchd.log</string>
</dict>
</plist>
EOF

# Load the launch agent
launchctl load ~/Library/LaunchAgents/com.mlx.talks.categorizer.plist
```

## Step 4: Verify the Cronjob Setup

1. **Check if crontab is active**:
   ```bash
   crontab -l
   ```

2. **Monitor the log file**:
   ```bash
   tail -f /Users/$(whoami)/Dev/mlx-talks-categorizer/cronjob.log
   ```

3. **Check system logs** (for troubleshooting):
   ```bash
   log show --predicate 'eventMessage contains "cron"' --info --last 1h
   ```

## Configuration Recommendations

### 1. Adjust Paths in config.json

Update your `config.json` with absolute paths for reliability:

```json
{
  "google_drive_path": "/Users/$(whoami)/Google Drive/Audio",
  "local_audio_path": "/Users/$(whoami)/Audio Hijack",
  "output_base_path": "/Users/$(whoami)/Dev/mlx-talks-categorizer/organized_talks"
}
```

### 2. Email Notifications (Optional)

Add email notifications to your wrapper script:

```bash
# Add this to the end of run_audio_manager.sh
if [ $? -eq 0 ]; then
    echo "Audio processing completed successfully at $(date)" | mail -s "MLX Audio Manager Success" your-email@example.com
else
    echo "Audio processing failed at $(date). Check logs." | mail -s "MLX Audio Manager Error" your-email@example.com
fi
```

## Troubleshooting

### Common Issues

1. **Permission Denied Errors**:
   - Ensure Full Disk Access is granted to Terminal
   - Check file permissions: `ls -la run_audio_manager.sh`
   - Make script executable: `chmod +x run_audio_manager.sh`

2. **Virtual Environment Not Found**:
   - Verify the virtual environment path in the script
   - Test activation manually: `source venv/bin/activate`

3. **PATH Issues**:
   - Add explicit PATH in the wrapper script
   - Use absolute paths for all binaries

4. **Cronjob Not Running**:
   - Check if cron service is running: `sudo launchctl list | grep cron`
   - Verify crontab syntax: Use online crontab validators
   - Check system logs for cron errors

5. **FFmpeg/Dependencies Not Found**:
   - Install via Homebrew: `brew install ffmpeg`
   - Add Homebrew paths to the script:
     ```bash
     export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
     ```

### Debugging Commands

```bash
# Check if cronjob is scheduled
crontab -l

# View recent log entries
tail -n 50 cronjob.log

# Test the script manually
./run_audio_manager.sh

# Check system cron logs
grep CRON /var/log/system.log

# For launchd (if using Option B)
launchctl list | grep com.mlx.talks.categorizer
```

## Security Considerations

1. **File Permissions**: Ensure the wrapper script and config files have appropriate permissions
2. **Log Rotation**: Consider setting up log rotation to prevent log files from growing too large
3. **Backup**: Regularly backup your configuration and organized files

## Example Log Monitoring

To continuously monitor the cronjob:

```bash
# Create a monitoring script
echo '#!/bin/bash
tail -f /Users/$(whoami)/Dev/mlx-talks-categorizer/cronjob.log | while read line; do
    echo "$(date): $line"
done' > monitor_cronjob.sh

chmod +x monitor_cronjob.sh
./monitor_cronjob.sh
```

## Stopping the Cronjob

### For crontab:
```bash
crontab -e
# Delete the line with the cronjob and save
```

### For launchd:
```bash
launchctl unload ~/Library/LaunchAgents/com.mlx.talks.categorizer.plist
rm ~/Library/LaunchAgents/com.mlx.talks.categorizer.plist
```

## Automated Setup Script Features

The `setup_cronjob.sh` script provides these schedule options:

1. **Daily at 2:00 AM** - Traditional off-hours processing
2. **Daily at 9:00 PM** - End-of-day processing when new files are likely added
3. **Every 6 hours** - Frequent processing for active workflows
4. **Weekly on Sunday at 3:00 AM** - Light processing for occasional use
5. **Every 2 hours** - For testing and development
6. **Custom schedule** - Enter your own cron expression

### Script Advantages

- **Automatic Path Detection**: Works regardless of installation location
- **Configuration Backup**: Creates `config.json.backup` before making changes
- **Virtual Environment Validation**: Tests dependencies before setup
- **Interactive Setup**: Guides you through all options
- **Cross-Method Support**: Handles both crontab and launchd
- **Monitoring Tools**: Creates log monitoring scripts automatically

### Generated Files

The setup script creates:
- `run_audio_manager.sh` - Main wrapper script with proper environment handling
- `monitor_cronjob.sh` - Real-time log monitoring utility
- `config.json.backup` - Backup of your original configuration
- Properly configured crontab entry or launchd plist file

---

**Note**: Replace `$(whoami)` with your actual username in all paths if the commands don't automatically expand it correctly.