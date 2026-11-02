# S8 middle band: predictions written BEFORE the synthetic tests (2026-10-21)
Exploratory first (heat16.py, where.py, shift.py, closeups): peaks follow content when shifted, but each unit has a
border prior (b7 col 0 floor; b8 col 13 negative; b9 rows 0/13 floor; b10 peaks on cols 0/13, rows 11-12).

| Neuron | One prediction (stimulus on flat mid-gray 128, 224 px) |
|---|---|
| b6 n1286 | A small sharp mark (dark or red, 12 px square or 6x40 px vertical bar) gives > 0.5 x max (1.76). The same mark blurred (sigma 3) gives < 0.25 x max. The centre of a 96 px block gives < 0.25 x max. |
| b7 n1155 | Upright filled dark triangle: map at the apex (top tip) > 2 x the map at the bottom tip of the same triangle turned upside down, and > 0.4 x max. |
| b8 n1026 | (colour, see test) a flat warm-brown textured field drives it negative; the same field in blue-grey (hue roll) is less negative by > 1.0. |
| b9 n64 | Dark rectangle on light gray: the band just above its BOTTOM edge is < -2 (negative pole); its top edge is > -1. Inverting the image moves the strongest negative to the top edge. |
| b10 n95 | Large dark disc (diameter 120) on gray: the inner lower-left / lower-right rim gives > 0.4 x max; a 30 px dark disc gives < 0.2 x max. |
