# sudo vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other -o uid=$(id -u) -o gid=$(id -g)
# cp -r /mnt/hgfs/sensus/* .

sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
    portaudio19-dev build-essential curl scrot
sudo apt install wmctrl xdotool pulseaudio-utils brightnessctl playerctl libgtk-3-bin
# Optional GTK overlay (--gui): PyGObject needs girepository-2.0.pc + GTK/WebKit typelibs
sudo apt install -y meson ninja-build pkg-config libglib2.0-dev libcairo2-dev \
    libgirepository-2.0-dev libgtk-4-dev libwebkitgtk-6.0-dev \
    gir1.2-gtk-4.0 gir1.2-webkit-6.0
# VMware HGFS (/mnt/hgfs/...) cannot create symlinks → venv needs --copies (above).
# WebKit bubblewrap sandbox often fails in VMware → overlay disables it by default (see overlay/app.py).
python3.11 -m venv --copies .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
sudo $(which playwright) install-deps
playwright install

# python -m sensus.voice.stt
# python -m sensus.voice.watson_stt

# write a wordle game in python using pygame

