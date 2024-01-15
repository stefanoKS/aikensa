#!/home/sekkei/Downloads/enter/envs/aikensa/bin/python


import sys
import os

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, 'YOLOv6'))

from aikensa.AIKensa import main
if __name__ == '__main__':
    main()