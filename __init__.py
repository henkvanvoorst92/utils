#init file for utils

__path__ = __import__('pkgutil').extend_path(__path__, __name__)
# print(__path__)
# import sys
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))