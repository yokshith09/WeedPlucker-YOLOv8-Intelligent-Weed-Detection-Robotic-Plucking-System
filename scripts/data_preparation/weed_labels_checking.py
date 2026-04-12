# Check how many weed labels you actually have
from pathlib import Path
n = sum(1 for f in Path(r'C:\NEWDRIVE\Model_train\dataset\balanced\labels\train').glob('*.txt')
        for line in f.read_text().splitlines()
        if line and int(line.split()[0]) == 0)
print('Weed annotations:', n)
