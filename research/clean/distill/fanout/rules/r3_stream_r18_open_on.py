"""STREAM round 3: ResNet stream-only, floor order, stream fold sfold=on, loop=open (rules/r3_stream_r18core.py). Input X only."""
from rules import r3_stream_r18core as c

r18 = c.make(order="floor", sets=("stream",), sfold="on", loop="open")
