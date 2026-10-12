"""Print which OS and Python this runs on. Used to check the Mac/Windows/Linux test bench."""
import platform
import sys

print(f"os={platform.system()} release={platform.release()} python={sys.version.split()[0]}")
