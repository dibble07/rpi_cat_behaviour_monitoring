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
1. Back up the external drive.
1. Check the drive name: `lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT /dev/sda`
1. Repartition the drive with one data partition and four 256 MiB swap partitions using explicit positive MiB boundaries:
```
DISK_MIB=$(sudo blockdev --getsize64 /dev/sda | awk '{print int($1/1024/1024)}')
P2_START=$((DISK_MIB-1024))
P3_START=$((DISK_MIB-768))
P4_START=$((DISK_MIB-512))
P5_START=$((DISK_MIB-256))

sudo parted -s /dev/sda unit MiB \
	mklabel gpt \
	mkpart primary ntfs 1 "$P2_START" \
	mkpart primary linux-swap "$P2_START" "$P3_START" \
	mkpart primary linux-swap "$P3_START" "$P4_START" \
	mkpart primary linux-swap "$P4_START" "$P5_START" \
	mkpart primary linux-swap "$P5_START" 100% \
	set 2 swap on \
	set 3 swap on \
	set 4 swap on \
	set 5 swap on
```
1. Re-read the partition table: `sudo partprobe /dev/sda`
1. Confirm the partition nodes exist: `sudo udevadm settle` then `lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT /dev/sda` and `sudo parted -s /dev/sda unit MiB print`
1. Create the data filesystem on `/dev/sda1` if needed. Example for NTFS: `sudo mkfs.ntfs -f /dev/sda1`
1. Find the data partition UUID: `sudo blkid /dev/sda1`
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
1. Find the swap partition UUIDs: `sudo blkid /dev/sda2 /dev/sda3 /dev/sda4 /dev/sda5`
1. Format each chunk as swap (`mkswap` supports one device per command):
```
sudo mkswap /dev/sda2
sudo mkswap /dev/sda3
sudo mkswap /dev/sda4
sudo mkswap /dev/sda5
```
1. Test activation once with descending priorities so later chunks are easiest to retire:
```
sudo swapon -p 40 /dev/sda2
sudo swapon -p 30 /dev/sda3
sudo swapon -p 20 /dev/sda4
sudo swapon -p 10 /dev/sda5
```
1. If the test worked, disable the chunks again before persisting them:
```
sudo swapoff /dev/sda5
sudo swapoff /dev/sda4
sudo swapoff /dev/sda3
sudo swapoff /dev/sda2
```
1. Add the swap partitions to `/etc/fstab` so systemd activates them automatically at boot:
```
UUID=<SWAP_UUID_2> none swap defaults,pri=40,nofail 0 0
UUID=<SWAP_UUID_3> none swap defaults,pri=30,nofail 0 0
UUID=<SWAP_UUID_4> none swap defaults,pri=20,nofail 0 0
UUID=<SWAP_UUID_5> none swap defaults,pri=10,nofail 0 0
```
1. Reload systemd and enable all swap entries immediately: `sudo systemctl daemon-reload` and `sudo swapon -a`
1. Verify zram stays above the HDD tier and the HDD chunks are visible individually: `swapon --show`

### Periodic chunk-wise reclaim
1. Create a reclaim script: `/usr/local/bin/reclaim-hdd-swap.sh`
1. Add content to file:
```
#!/bin/bash
set -euo pipefail

MIN_AVAILABLE_KB=$((700 * 1024))
SWAP_TARGETS=(
	"/dev/sda5:10"
	"/dev/sda4:20"
	"/dev/sda3:30"
	"/dev/sda2:40"
)

available_kb="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
if (( available_kb < MIN_AVAILABLE_KB )); then
	logger -t hdd-swap-reclaim "skip: MemAvailable=${available_kb}kB"
	exit 0
fi

for entry in "${SWAP_TARGETS[@]}"; do
	dev="${entry%%:*}"
	prio="${entry##*:}"
	used_kb="$(awk -v dev="$dev" '$1 == dev {print $4}' /proc/swaps)"

	if [[ -n "$used_kb" && "$used_kb" -gt 0 ]]; then
		if /sbin/swapoff "$dev"; then
			/sbin/swapon -p "$prio" "$dev"
			logger -t hdd-swap-reclaim "reclaimed chunk=$dev used_kb=${used_kb}"
		else
			logger -t hdd-swap-reclaim "swapoff failed chunk=$dev used_kb=${used_kb}"
		fi
		exit 0
	fi
done

logger -t hdd-swap-reclaim "skip: no used HDD swap chunks"
```
1. Make it executable: `sudo chmod +x /usr/local/bin/reclaim-hdd-swap.sh`
1. Create a oneshot service file: `/etc/systemd/system/hdd-swap-reclaim.service`
1. Add content to file:
```
[Unit]
Description=Reclaim one HDD swap chunk

[Service]
Type=oneshot
ExecStart=/usr/local/bin/reclaim-hdd-swap.sh
```
1. Create a timer file: `/etc/systemd/system/hdd-swap-reclaim.timer`
1. Add content to file:
```
[Unit]
Description=Attempt chunk-wise HDD swap reclaim

[Timer]
OnBootSec=20min
OnUnitActiveSec=3min
Persistent=true

[Install]
WantedBy=timers.target
```
1. Reload manager and enable the timer: `sudo systemctl daemon-reload` and `sudo systemctl enable --now hdd-swap-reclaim.timer`
1. Manually test one reclaim run: `sudo systemctl start hdd-swap-reclaim.service`
1. Watch reclaim logs: `journalctl -u hdd-swap-reclaim.service -f`