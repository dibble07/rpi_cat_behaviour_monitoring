# rpi_cat_behaviour_monitoring
Raspberry Pi setup for app

## Disable wifi powersave mode
1. Create a systemd service file: `/etc/systemd/system/disable-wifi-powersave.service`
1. Add content to file:
```
[Unit]
Description=Disable WiFi power saving
After=network-pre.target
Wants=network-pre.target

[Service]
Type=oneshot
ExecStart=/usr/sbin/iw dev wlan0 set power_save off
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```
1. Reload manager and enable service: `sudo systemctl daemon-reload` and `sudo systemctl enable disable-wifi-powersave.service`
1. Optionally run script immediately to test: `sudo systemctl start disable-wifi-powersave.service`
1. Check output `iw dev wlan0 get power_save`

## Python env setup
`picamera2` depends on system libraries not available via pip. Install via apt and create a venv with system site-packages access:
```
sudo apt install -y python3-picamera2 python3-libcamera
uv venv --python 3.13 --system-site-packages
uv sync --no-dev
```

## Startup script
1. Create script file: `/home/rpdibble/rpi_cat_behaviour_monitoring.sh`
1. Add content to this file:
```
#!/bin/bash
set -e
git reset --hard
git fetch origin
git checkout test
git reset --hard origin/test
/home/rpdibble/.local/bin/uv sync --no-dev
.venv/bin/python src/app.py
```
1. Make it executable: `sudo chmod +x /home/rpdibble/rpi_cat_behaviour_monitoring.sh`
1. Create a systemd service file: `/etc/systemd/system/startup.service`
1. Add content to file:
```
[Unit]
Description=Run startup script for cat behaviour monitor
After=network-online.target
Wants=network-online.target

[Service]
User=rpdibble
WorkingDirectory=/home/rpdibble/rpi_cat_behaviour_monitoring
ExecStart=/home/rpdibble/rpi_cat_behaviour_monitoring.sh
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=30
KillSignal=SIGINT
TimeoutStopSec=30
StartLimitAction=none

[Install]
WantedBy=multi-user.target
```
1. Reload manager and enable service: `sudo systemctl daemon-reload` and `sudo systemctl enable startup.service`
1. Optionally run script immediately to test: `sudo systemctl start startup.service`
1. Check service status: `sudo systemctl status startup.service`
1. View logs: `journalctl -u startup.service -b -o short-precise`
1. Stop running service and disable startup execution: `sudo systemctl stop startup.service` and `sudo systemctl disable startup.service`

### Scheduled start/stop
Uses two timers to start and stop the service on a schedule.

1. Create a timer to start the service: `/etc/systemd/system/startup.timer`
1. Add content to file:
```
[Unit]
Description=Start cat behaviour monitor on schedule

[Timer]
OnCalendar=*-*-* 04:00:00
Persistent=true

[Install]
WantedBy=timers.target
```
1. Create a oneshot service to stop the monitor: `/etc/systemd/system/startup-stop.service`
1. Add content to file:
```
[Unit]
Description=Stop cat behaviour monitor

[Service]
Type=oneshot
ExecStart=/usr/bin/systemctl stop startup.service
```
1. Create a timer to stop the service: `/etc/systemd/system/startup-stop.timer`
1. Add content to file:
```
[Unit]
Description=Stop cat behaviour monitor on schedule

[Timer]
OnCalendar=*-*-* 23:59:00
Persistent=true

[Install]
WantedBy=timers.target
```
1. Reload manager, enable timers, disable direct boot start: `sudo systemctl daemon-reload` and `sudo systemctl enable --now startup.timer startup-stop.timer` and `sudo systemctl disable startup.service`
1. Verify: `systemctl list-timers startup.timer startup-stop.timer`
1. To manually start outside the window: `sudo systemctl start startup.service`

## Cloud sync script
1. [Install rclone](https://rclone.org/install/#script-installation)
1. Run `rclone config` and get credentials [info](https://console.cloud.google.com/auth/clients?project=rpi-cat-behaviour-monitor) to complete
1. Create a systemd service file: `/etc/systemd/system/rclone-sync.service`
1. Add content to file:
```
[Unit]
Description=Sync object_clips folder to Google Drive

[Service]
Type=oneshot
User=rpdibble
ExecStart=/usr/bin/rclone copy --ignore-checksum --ignore-size /home/rpdibble/rpi_cat_behaviour_monitoring/object_clips/ gdrive:rpi_cat_behaviour_monitoring/object_clips
ExecStart=-/usr/bin/rclone copy /mnt/hdd/object_clips/ gdrive:rpi_cat_behaviour_monitoring/object_clips
```
1. Create a systemd timer file: `/etc/systemd/system/rclone-sync.timer`
1. Add content to file:
```
[Unit]
Description=Run rclone sync every minute

[Timer]
OnBootSec=10min
OnUnitActiveSec=10min
Persistent=true

[Install]
WantedBy=timers.target
```
1. Reload manager and enable service: `sudo systemctl daemon-reload` and `sudo systemctl enable rclone-sync.timer`
1. Optionally start service immediately: `sudo systemctl start rclone-sync.timer`
1. Manually trigger clone: `sudo systemctl start rclone-sync.service`
1. List timers: `systemctl list-timers rclone-sync.timer`
1. View logs: `journalctl -u rclone-sync.service -f`

## External HDD mount
1. Find the drive's UUID: `sudo blkid /dev/sda1`
1. Create the mount point directory: `sudo mkdir -p /mnt/hdd`
1. Create a systemd mount unit file: `/etc/systemd/system/mnt-hdd.mount`
1. Add content to file (replace `<UUID>` with the value from `blkid`, and set `Type` to match your filesystem):
```
[Unit]
Description=Seagate Expansion Drive
After=local-fs-pre.target
Requires=local-fs-pre.target

[Mount]
What=UUID=AC4EE3984EE35A1A
Where=/mnt/hdd
Type=ntfs-3g
Options=defaults,uid=1000,gid=1000,umask=0022,nofail

[Install]
WantedBy=multi-user.target
```
1. Create a systemd automount unit file: `/etc/systemd/system/mnt-hdd.automount`
1. Add content to file:
```
[Unit]
Description=Automount Seagate Expansion Drive
After=local-fs-pre.target

[Automount]
Where=/mnt/hdd
TimeoutIdleSec=0

[Install]
WantedBy=multi-user.target
```
1. Install `ntfs-3g` if not already present (skip if using exfat): `sudo apt install -y ntfs-3g`
1. Reload manager and enable the automount: `sudo systemctl daemon-reload` and `sudo systemctl enable mnt-hdd.automount`
1. Start it immediately: `sudo systemctl start mnt-hdd.automount`
1. Verify it mounts on access: `ls /mnt/hdd`
1. Check status: `sudo systemctl status mnt-hdd.mount`

### Memory swap on HDD
Current state is zram-only (`/dev/zram0`). Keep zram enabled and add HDD swap as a secondary tier to improve behavior when memory pressure spikes.

1. Create a swap file on the HDD: `sudo dd if=/dev/zero of=/mnt/hdd/swapfile bs=1M count=2048 status=progress`
1. Set correct permissions: `sudo chmod 600 /mnt/hdd/swapfile`
1. Set root ownership: `sudo chown root:root /mnt/hdd/swapfile`
1. Format as swap: `sudo mkswap /mnt/hdd/swapfile`
1. Test activation once: `sudo swapon -p 50 /mnt/hdd/swapfile`
1. If the test worked, disable it again before wiring it into systemd: `sudo swapoff /mnt/hdd/swapfile`
1. Create a systemd service file to activate swap after the HDD mounts: `/etc/systemd/system/hdd-swap.service`
1. Add content to file:
```
[Unit]
Description=Enable swap on HDD
After=mnt-hdd.mount
Requires=mnt-hdd.mount

[Service]
Type=oneshot
ExecStart=/sbin/swapon -p 50 /mnt/hdd/swapfile
ExecStop=/sbin/swapoff /mnt/hdd/swapfile
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```
1. Reload manager and enable service: `sudo systemctl daemon-reload` and `sudo systemctl enable hdd-swap.service`
1. Start immediately: `sudo systemctl start hdd-swap.service`
1. Verify both tiers are active and priority is correct: `swapon --show`
1. Optional runtime check: `free -h`