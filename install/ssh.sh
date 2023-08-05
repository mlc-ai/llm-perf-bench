set -euxo pipefail

mkdir /var/run/sshd
echo 'root:llm_perf' | chpasswd
sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
sed -ri 's/^#?X11UseLocalhost\s+.*/X11UseLocalhost no/' /etc/ssh/sshd_config
sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
