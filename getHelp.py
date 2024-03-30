import subprocess, os

commands = ['alias', 'apropos', 'apt', 'apt-get', 'aptitude', 'aspell', 'at', 'awk', 'basename', 'base32', 'base64', 'bash', 'bc', 'bg', 'bind', 'break', 'builtin', 'bzip2', 'cal', 'caller', 'case', 'cat', 'cd', 'cfdisk', 'chattr', 'chgrp', 'chmod', 'chwon', 'chpasswd', 'chroot', 'chkconfig', 'cksum', 'clear', 'clear_console', 'cmp', 'comm', 'command', 'continue', 'cp', 'cpio', 'cron', 'crontab', 'csplit', 'curl', 'cut', 'date', 'dc', 'dd', 'ddrescue', 'declare', 'df', 'diff', 'diff3', 'dig', 'dir', 'dircolors', 'dirname', 'dirs', 'dos2unix', 'dmesg', 'dpkg', 'du', 'echo', 'egrep', 'eject', 'enable', 'env', 'ethtool', 'eval', 'exec', 'exit', 'expand', 'export', 'expr', 'false', 'fdformat', 'fdisk', 'fg', 'fgrep', 'file', 'find', 'fmt', 'fold', 'for', 'format', 'free', 'fsck', 'ftp', 'function', 'fuser', 'gawk', 'getopts', 'getfact', 'grep', 'groupadd', 'groupdel', 'groupmod', 'groups', 'gzip', 'hash', 'head', 'help', 'history', 'hostname', 'htop', 'iconv', 'id', 'if', 'ifconfig', 'ifdown', 'ifup', 'import', 'instlal', 'iostat', 'ip']


for command in commands:
    try:
        subprocess.check_output(f"{command} --help", shell=True, stderr=subprocess.STDOUT)
        os.system(f"{command} --help > help-docs/{command}.txt")
    except subprocess.CalledProcessError as e:
        print(f"{command} didn't work")

