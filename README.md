# rpi_cat_behaviour_monitoring
Raspberry Pi app to monitor cats with live video feed and identify poor behaviour

## Startup script
1. Create your script file: `/home/robertdibble/rpi_cat_behaviour_monitoring.sh`
1. Make it executable: `chmod +x /home/robertdibble/rpi_cat_behaviour_monitoring.sh`
1. Create a systemd service file: `/etc/systemd/system/startup.service`
1. Add content to file:
```
[Unit]
Description=Run startup script for cat behaviour monitor
After=network.target

[Service]
User=robertdibble
WorkingDirectory=/home/robertdibble/rpi_cat_behaviour_monitoring
ExecStart=/home/robertdibble/rpi_cat_behaviour_monitoring.sh
Restart=on-failure
RestartSec=10
StartLimitBurst=3

[Install]
WantedBy=multi-user.target
```
1. Reload manager and enable service: `sudo systemctl daemon-reload` and `sudo systemctl enable startup.service`
1. Optionally run script immediately to test: `sudo systemctl start startup.service`
1. Check service status: `sudo systemctl status startup.service`
1. View logs: `journalctl -u startup.service`
1. Stop running service and disable startup execution: `sudo systemctl stop startup.service` and `sudo systemctl disable startup.service`