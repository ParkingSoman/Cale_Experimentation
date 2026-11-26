# Small system: read and write cards

Reads: z vs random reads (floor: median 2.6, 95th 3.4). Writes: raw cosine (s4b0 floor: median .13, 95th .16).
A's real block-0 writes: top cosine median .25; real read kernels: top z median 3.8 (README K5b).

| Neuron | Read: top 3 up | Read: top 2 down | Read leftover | Write: top 3 up | Write: top 2 down | Write leftover |
|---|---|---|---|---|---|---|
| parent s3:69 | faucet +6.58, fan +5.63, clock +5.42 | concrete -1.36, wood -1.27 | 0.40 | faucet +0.42, fan +0.35, clock +0.34 | concrete -0.09, wood -0.08 | 0.70 |
| OCN | fan +6.67, wheel +6.20, chandelier +5.79 | concrete -2.29, brick -1.92 | 0.73 | zigzagged +0.41, chequered +0.35, studded +0.31 | matted -0.39, flecked -0.39 | 0.68 |
| child n305 | shoe_shop-s +2.92, paper +2.68, painting +2.68 | skin -3.97, freckled -3.54 | 0.85 | zigzagged +0.18, waffled +0.16, childs_room-s +0.15 | pitted -0.20, blotchy -0.18 | 0.87 |
| child n305 condition | glass +2.73, cracked +2.66, paper +2.43 | skin -4.12, freckled -3.68 | 0.85 |  | | |
| child n97 | ice rink +3.78, fence +3.72, airport_terminal-s +3.35 | paisley -3.29, tent -3.29 | 0.83 | ice rink +0.18, chequered +0.17, airport_terminal-s +0.16 | forest-needleleaf-s -0.15, alley-s -0.14 | 0.85 |
| child n97 condition | fence +3.96, ice rink +3.71, airport_terminal-s +3.48 | paisley -3.63, motorbike -3.56 | 0.81 |  | | |
| child n23 | pitted +6.23, porous +5.32, blotchy +5.00 | swirly -4.92, interlaced -4.28 | 0.80 | art_gallery-s +0.16, dome +0.15, shade +0.14 | black-c -0.15, fibrous -0.14 | 0.90 |
| child n23 condition | pitted +6.40, porous +5.52, blotchy +5.10 | swirly -5.26, interlaced -4.53 | 0.79 |  | | |
| child n163 | sidewalk +3.36, floor +3.24, museum-indoor-s +3.16 | casino-indoor-s -3.00, awning -2.76 | 0.81 | art_gallery-s +0.22, laminate +0.19, poolroom-home-s +0.18 | trench -0.16, skyscraper-s -0.14 | 0.85 |
| child n163 condition | sidewalk +3.39, floor +3.31, carpet +3.02 | casino-indoor-s -3.09, headlight -2.98 | 0.80 |  | | |
| child n385 | chandelier +4.12, vase +3.74, pitted +3.62 | striped -4.42, chequered -3.73 | 0.82 | bar +0.23, ring +0.18, bar-s +0.17 | rock -0.17, matted -0.15 | 0.86 |
| child n385 condition | pitted +3.75, boat +3.45, chandelier +3.42 | striped -4.41, chequered -3.79 | 0.83 |  | | |
| child n297 | stained +4.20, blotchy +4.17, smeared +4.13 | perforated -3.77, meshed -3.17 | 0.81 | paisley +0.30, poster +0.29, trade name +0.28 | grooved -0.19, basement-s -0.17 | 0.79 |
| child n297 condition | blotchy +4.33, marbled +4.32, stained +4.31 | perforated -3.90, meshed -3.20 | 0.80 |  | | |
| child n464 | interlaced +3.74, paisley +3.04, food +2.86 | dishwasher -4.16, chequered -4.16 | 0.77 | zigzagged +0.30, interlaced +0.28, spiralled +0.25 | creek-s -0.28, pitted -0.26 | 0.81 |
| child n464 condition | interlaced +3.68, paisley +2.93, creek-s +2.58 | dishwasher -4.18, chequered -4.15 | 0.78 |  | | |
| child n237 | lacelike +4.60, sprinkled +4.24, amusement_park-s +3.79 | cradle -4.41, radiator -4.32 | 0.80 | polka-dotted +0.27, dotted +0.26, studded +0.26 | pitted -0.22, cracked -0.22 | 0.81 |
| child n237 condition | lacelike +4.58, sprinkled +4.13, amusement_park-s +3.79 | cradle -4.53, radiator -4.34 | 0.79 |  | | |
| child n108 | wheel +3.25, motorbike +2.69, bicycle +2.52 | chequered -2.18, dishwasher -2.16 | 0.91 | car +0.23, wheel +0.23, truck +0.22 | flecked -0.21, ocean-s -0.20 | 0.84 |
| child n108 condition | wheel +2.59, motorbike +2.30, bicycle +2.00 | chequered -2.16, dishwasher -2.15 | 0.93 |  | | |
| child n510 | bubbly +4.88, paisley +4.80, bumpy +4.41 | plaza-s -4.01, apartment_building-outdoor-s -3.96 | 0.80 | studded +0.25, paisley +0.23, swirly +0.19 | matted -0.26, fibrous -0.23 | 0.81 |
| child n510 condition | paisley +4.72, bubbly +4.72, bumpy +4.39 | plaza-s -4.04, apartment_building-outdoor-s -3.84 | 0.79 |  | | |
| child n275 | chandelier +5.22, fan +4.31, flower +3.20 | spiralled -4.06, eye -3.70 | 0.83 | zigzagged +0.45, chequered +0.35, interlaced +0.27 | potholed -0.31, marbled -0.30 | 0.69 |
| child n275 condition | chandelier +4.62, fan +3.62, zigzagged +2.96 | spiralled -4.14, eye -3.92 | 0.84 |  | | |

## Cards

# Read: parent s3:69

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| faucet | part | 6.575 |
| fan | object | 5.635 |
| clock | object | 5.421 |
| chandelier | object | 4.329 |
| wheel | part | 4.22 |
| minibike | object | 4.062 |
| nose | part | 3.977 |
| chain wheel | part | 3.672 |

**down**

| concept | category | value |
|---|---|---|
| concrete | material | -1.365 |
| wood | material | -1.274 |
| brick | material | -1.214 |
| granite | material | -1.017 |
| balcony-interior-s | scene | -1.001 |
| cage / bird cage | object | -0.978 |
| laminate | material | -0.893 |
| corridor-s | scene | -0.867 |

**decomposition**

| concept | category | value |
|---|---|---|
| faucet | part | 0.522 |
| clock | object | 0.342 |
| flower | object | 0.198 |
| nose | part | 0.138 |
| fire escape | part | -0.131 |

**leftover**: 0.399

**examples**

| concept | category | value |
|---|---|---|
| wheel | part | 0.027 |
| car | object | 0.02 |
| head | part | 0.017 |
| cat | object | 0.013 |
| dog | object | 0.012 |

# Write: parent s3:69

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| faucet | part | 0.418 |
| fan | object | 0.351 |
| clock | object | 0.342 |
| chandelier | object | 0.274 |
| wheel | part | 0.267 |
| nose | part | 0.25 |
| minibike | object | 0.249 |
| chain wheel | part | 0.228 |

**down**

| concept | category | value |
|---|---|---|
| concrete | material | -0.085 |
| wood | material | -0.076 |
| brick | material | -0.072 |
| granite | material | -0.065 |
| balcony-interior-s | scene | -0.062 |
| cage / bird cage | object | -0.059 |
| laminate | material | -0.056 |
| painted | material | -0.055 |

**decomposition**

| concept | category | value |
|---|---|---|
| faucet | part | 0.584 |
| rocking chair | object | -0.282 |
| wood | material | 0.175 |
| clock | object | 0.272 |
| cap | part | -0.218 |

**leftover**: 0.7

**examples**

| concept | category | value |
|---|---|---|
| wheel | part | 0.027 |
| car | object | 0.02 |
| head | part | 0.017 |
| cat | object | 0.013 |
| dog | object | 0.012 |

# Read: OCN

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| fan | object | 6.675 |
| wheel | part | 6.201 |
| chandelier | object | 5.791 |
| bicycle | object | 4.933 |
| nose | part | 4.525 |
| plaything | object | 4.195 |
| flower | object | 3.892 |
| motorbike | object | 3.869 |

**down**

| concept | category | value |
|---|---|---|
| concrete | material | -2.294 |
| brick | material | -1.916 |
| cage / bird cage | object | -1.842 |
| wood | material | -1.704 |
| apartment_building-outdoor-s | scene | -1.553 |
| corridor-s | scene | -1.513 |
| alley-s | scene | -1.477 |
| skyscraper-s | scene | -1.437 |

**decomposition**

| concept | category | value |
|---|---|---|
| wheel | part | 0.282 |
| fan | object | 0.222 |
| plaything | object | 0.195 |
| head | part | 0.162 |
| coffee table | object | 0.154 |

**leftover**: 0.733

**examples**

| concept | category | value |
|---|---|---|
| wheel | part | 0.037 |
| head | part | 0.026 |
| car | object | 0.026 |
| cat | object | 0.023 |
| bicycle | object | 0.017 |

# Write: OCN

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.412 |
| chequered | texture | 0.353 |
| studded | texture | 0.307 |
| polka-dotted | texture | 0.304 |
| interlaced | texture | 0.283 |
| dotted | texture | 0.282 |
| museum-indoor-s | scene | 0.282 |
| art_gallery-s | scene | 0.267 |

**down**

| concept | category | value |
|---|---|---|
| matted | texture | -0.392 |
| flecked | texture | -0.385 |
| fibrous | texture | -0.375 |
| pitted | texture | -0.372 |
| marbled | texture | -0.351 |
| porous | texture | -0.347 |
| blotchy | texture | -0.336 |
| cracked | texture | -0.32 |

**decomposition**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.319 |
| studded | texture | 0.262 |
| jacuzzi-indoor-s | scene | 0.148 |
| gymnasium-indoor-s | scene | -0.199 |
| museum-indoor-s | scene | 0.174 |

**leftover**: 0.682

**examples**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.076 |
| chequered | texture | 0.049 |
| head | part | 0.025 |
| dog | object | 0.024 |
| cat | object | 0.02 |

# Read: child n305

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| shoe_shop-s | scene | 2.919 |
| paper | material | 2.682 |
| painting | object | 2.681 |
| glass | material | 2.619 |
| cracked | texture | 2.533 |
| poster | object | 2.42 |
| shelf | object | 2.21 |
| fan | object | 2.193 |

**down**

| concept | category | value |
|---|---|---|
| skin | material | -3.974 |
| freckled | texture | -3.544 |
| ball | object | -3.247 |
| jacuzzi-indoor-s | scene | -3.008 |
| arm | part | -2.758 |
| bed | object | -2.672 |
| spiralled | texture | -2.626 |
| staircase-s | scene | -2.6 |

**decomposition**

| concept | category | value |
|---|---|---|
| person | object | -0.224 |
| painting | object | 0.193 |
| dinette-vehicle-s | scene | -0.194 |
| shoe_shop-s | scene | 0.172 |
| wallpaper | material | -0.138 |

**leftover**: 0.849

**examples**

| concept | category | value |
|---|---|---|
| head | part | 0.028 |
| cat | object | 0.019 |
| painting | object | 0.019 |
| white-c | color | 0.014 |
| dog | object | 0.012 |

# Write: child n305

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.184 |
| waffled | texture | 0.157 |
| childs_room-s | scene | 0.154 |
| nursery-s | scene | 0.154 |
| chequered | texture | 0.153 |
| pantry | object | 0.152 |
| file cabinet | object | 0.152 |
| playroom-s | scene | 0.151 |

**down**

| concept | category | value |
|---|---|---|
| pitted | texture | -0.197 |
| blotchy | texture | -0.183 |
| bridge-s | scene | -0.175 |
| porous | texture | -0.166 |
| flecked | texture | -0.165 |
| marbled | texture | -0.161 |
| forest-broadleaf-s | scene | -0.156 |
| bubbly | texture | -0.153 |

**decomposition**

| concept | category | value |
|---|---|---|
| pitted | texture | -0.183 |
| bridge-s | scene | -0.242 |
| lobby-s | scene | -0.181 |
| cat | object | 0.284 |
| muzzle | part | -0.249 |

**leftover**: 0.869

**examples**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.073 |
| cat | object | 0.044 |
| torso | part | 0.032 |
| head | part | 0.03 |
| dog | object | 0.028 |

# Read: child n305 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| glass | material | 2.732 |
| cracked | texture | 2.661 |
| paper | material | 2.431 |
| shoe_shop-s | scene | 2.42 |
| painting | object | 2.38 |
| monitor | part | 2.209 |
| plaza-s | scene | 2.199 |
| castle-s | scene | 2.16 |

**down**

| concept | category | value |
|---|---|---|
| skin | material | -4.121 |
| freckled | texture | -3.683 |
| ball | object | -3.379 |
| jacuzzi-indoor-s | scene | -3.352 |
| arm | part | -2.895 |
| spiralled | texture | -2.867 |
| bed | object | -2.765 |
| person | object | -2.618 |

**decomposition**

| concept | category | value |
|---|---|---|
| person | object | -0.219 |
| bed | object | -0.173 |
| paper | material | 0.165 |
| perforated | texture | -0.14 |
| staircase-s | scene | -0.134 |

**leftover**: 0.854

**examples**

| concept | category | value |
|---|---|---|
| head | part | 0.022 |
| painting | object | 0.019 |
| white-c | color | 0.017 |
| cat | object | 0.015 |
| windowpane | object | 0.012 |

# Read: child n97

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| ice rink | object | 3.779 |
| fence | object | 3.719 |
| airport_terminal-s | scene | 3.345 |
| cradle | object | 3.209 |
| plaza-s | scene | 3.196 |
| nose | part | 3.123 |
| auditorium-s | scene | 2.752 |
| railing | object | 2.662 |

**down**

| concept | category | value |
|---|---|---|
| paisley | texture | -3.294 |
| tent | object | -3.285 |
| torso | part | -3.088 |
| jacket | object | -3.022 |
| horse | object | -3.007 |
| canopy | part | -2.94 |
| pottedplant | object | -2.938 |
| arm | part | -2.921 |

**decomposition**

| concept | category | value |
|---|---|---|
| fence | object | 0.246 |
| hovel | object | -0.174 |
| arm | part | -0.166 |
| ice rink | object | 0.155 |
| paisley | texture | -0.143 |

**leftover**: 0.833

**examples**

| concept | category | value |
|---|---|---|
| head | part | 0.025 |
| cat | object | 0.021 |
| chequered | texture | 0.02 |
| ceiling | object | 0.014 |
| dog | object | 0.013 |

# Write: child n97

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| ice rink | object | 0.184 |
| chequered | texture | 0.168 |
| airport_terminal-s | scene | 0.162 |
| auditorium-s | scene | 0.155 |
| baggage_claim-s | scene | 0.152 |
| cradle | object | 0.148 |
| bannister | object | 0.141 |
| jacuzzi-indoor-s | scene | 0.14 |

**down**

| concept | category | value |
|---|---|---|
| forest-needleleaf-s | scene | -0.149 |
| alley-s | scene | -0.137 |
| knitted | texture | -0.135 |
| forest-broadleaf-s | scene | -0.132 |
| lacelike | texture | -0.131 |
| forest_path-s | scene | -0.126 |
| trench | object | -0.12 |
| crosshatched | texture | -0.116 |

**decomposition**

| concept | category | value |
|---|---|---|
| ice rink | object | 0.365 |
| water | object | -0.292 |
| towel | object | -0.17 |
| chequered | texture | 0.215 |
| sprinkled | texture | 0.172 |

**leftover**: 0.847

**examples**

| concept | category | value |
|---|---|---|
| chequered | texture | 0.045 |
| head | part | 0.044 |
| cat | object | 0.039 |
| dog | object | 0.022 |
| car | object | 0.015 |

# Read: child n97 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| fence | object | 3.959 |
| ice rink | object | 3.71 |
| airport_terminal-s | scene | 3.479 |
| plaza-s | scene | 3.321 |
| cradle | object | 3.292 |
| cage / bird cage | object | 2.907 |
| auditorium-s | scene | 2.889 |
| railing | object | 2.843 |

**down**

| concept | category | value |
|---|---|---|
| paisley | texture | -3.627 |
| motorbike | object | -3.562 |
| pottedplant | object | -3.396 |
| tent | object | -3.371 |
| jacket | object | -3.176 |
| horse | object | -3.154 |
| torso | part | -3.124 |
| leg | part | -3.11 |

**decomposition**

| concept | category | value |
|---|---|---|
| fence | object | 0.206 |
| motorbike | object | -0.194 |
| person | object | -0.183 |
| airport_terminal-s | scene | 0.173 |
| canopy | part | -0.151 |

**leftover**: 0.809

**examples**

| concept | category | value |
|---|---|---|
| chequered | texture | 0.028 |
| head | part | 0.019 |
| cat | object | 0.017 |
| building | object | 0.015 |
| white-c | color | 0.013 |

# Read: child n23

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| pitted | texture | 6.228 |
| porous | texture | 5.324 |
| blotchy | texture | 5.002 |
| stained | texture | 4.464 |
| sprinkled | texture | 4.217 |
| potholed | texture | 3.868 |
| flecked | texture | 3.704 |
| marbled | texture | 3.64 |

**down**

| concept | category | value |
|---|---|---|
| swirly | texture | -4.915 |
| interlaced | texture | -4.28 |
| paisley | texture | -4.139 |
| vault | object | -3.586 |
| meshed | texture | -3.35 |
| awning | object | -3.269 |
| trade name | object | -3.153 |
| bicycle | object | -3.09 |

**decomposition**

| concept | category | value |
|---|---|---|
| blotchy | texture | 0.27 |
| castle-s | scene | 0.2 |
| hair | part | 0.153 |
| metal | material | -0.155 |
| paisley | texture | -0.143 |

**leftover**: 0.798

**examples**

| concept | category | value |
|---|---|---|
| person | object | 0.043 |
| head | part | 0.04 |
| hair | part | 0.022 |
| dog | object | 0.018 |
| brown-c | color | 0.014 |

# Write: child n23

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| art_gallery-s | scene | 0.158 |
| dome | part | 0.146 |
| shade | part | 0.142 |
| white-c | color | 0.136 |
| pink-c | color | 0.131 |
| perforated | texture | 0.127 |
| chequered | texture | 0.125 |
| yellow-c | color | 0.119 |

**down**

| concept | category | value |
|---|---|---|
| black-c | color | -0.152 |
| fibrous | texture | -0.141 |
| veined | texture | -0.132 |
| matted | texture | -0.131 |
| foot | part | -0.111 |
| aircraft carrier | object | -0.111 |
| smeared | texture | -0.104 |
| guitar | object | -0.104 |

**decomposition**

| concept | category | value |
|---|---|---|
| art_gallery-s | scene | 0.194 |
| dome | part | 0.146 |
| porous | texture | 0.205 |
| fibrous | texture | -0.174 |
| loudspeaker | object | -0.125 |

**leftover**: 0.898

**examples**

| concept | category | value |
|---|---|---|
| person | object | 0.051 |
| torso | part | 0.024 |
| head | part | 0.024 |
| zigzagged | texture | 0.019 |
| dog | object | 0.018 |

# Read: child n23 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| pitted | texture | 6.404 |
| porous | texture | 5.516 |
| blotchy | texture | 5.1 |
| stained | texture | 4.497 |
| sprinkled | texture | 4.051 |
| potholed | texture | 3.968 |
| flecked | texture | 3.92 |
| marbled | texture | 3.765 |

**down**

| concept | category | value |
|---|---|---|
| swirly | texture | -5.264 |
| interlaced | texture | -4.534 |
| paisley | texture | -4.467 |
| bicycle | object | -3.911 |
| vault | object | -3.664 |
| motorbike | object | -3.474 |
| headlight | part | -3.339 |
| trade name | object | -3.336 |

**decomposition**

| concept | category | value |
|---|---|---|
| blotchy | texture | 0.288 |
| castle-s | scene | 0.179 |
| bicycle | object | -0.151 |
| hair | part | 0.145 |
| bus | object | -0.144 |

**leftover**: 0.787

**examples**

| concept | category | value |
|---|---|---|
| person | object | 0.047 |
| head | part | 0.038 |
| hair | part | 0.023 |
| dog | object | 0.017 |
| brown-c | color | 0.014 |

# Read: child n163

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| sidewalk | object | 3.361 |
| floor | object | 3.243 |
| museum-indoor-s | scene | 3.16 |
| court | object | 2.927 |
| carpet | material | 2.919 |
| fan | object | 2.661 |
| ice rink | object | 2.65 |
| bathtub | object | 2.617 |

**down**

| concept | category | value |
|---|---|---|
| casino-indoor-s | scene | -3.001 |
| awning | object | -2.76 |
| building | object | -2.608 |
| balcony | part | -2.463 |
| saddle | part | -2.455 |
| windowpane | object | -2.377 |
| headlight | part | -2.354 |
| slot machine | object | -2.329 |

**decomposition**

| concept | category | value |
|---|---|---|
| floor | object | 0.293 |
| casino-indoor-s | scene | -0.192 |
| counter | object | 0.159 |
| windowpane | object | -0.133 |
| conference_room-s | scene | -0.13 |

**leftover**: 0.811

**examples**

| concept | category | value |
|---|---|---|
| floor | object | 0.027 |
| head | part | 0.017 |
| road | object | 0.017 |
| sidewalk | object | 0.014 |
| cat | object | 0.014 |

# Write: child n163

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| art_gallery-s | scene | 0.224 |
| laminate | material | 0.19 |
| poolroom-home-s | scene | 0.181 |
| bathtub | object | 0.18 |
| ice rink | object | 0.164 |
| museum-indoor-s | scene | 0.163 |
| pink-c | color | 0.157 |
| painted | material | 0.154 |

**down**

| concept | category | value |
|---|---|---|
| trench | object | -0.156 |
| skyscraper-s | scene | -0.143 |
| pottedplant | object | -0.14 |
| fibrous | texture | -0.132 |
| basket | object | -0.131 |
| black-c | color | -0.13 |
| casino-indoor-s | scene | -0.13 |
| bakery-shop-s | scene | -0.125 |

**decomposition**

| concept | category | value |
|---|---|---|
| art_gallery-s | scene | 0.348 |
| snow | object | 0.244 |
| pool table | object | 0.276 |
| tree | object | 0.273 |
| fibrous | texture | -0.132 |

**leftover**: 0.846

**examples**

| concept | category | value |
|---|---|---|
| ceiling | object | 0.03 |
| bed | object | 0.018 |
| white-c | color | 0.017 |
| grey-c | color | 0.017 |
| bedroom-s | scene | 0.017 |

# Read: child n163 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| sidewalk | object | 3.39 |
| floor | object | 3.309 |
| carpet | material | 3.019 |
| museum-indoor-s | scene | 2.939 |
| court | object | 2.791 |
| tile | material | 2.785 |
| ground | object | 2.549 |
| laminate | material | 2.541 |

**down**

| concept | category | value |
|---|---|---|
| casino-indoor-s | scene | -3.09 |
| headlight | part | -2.975 |
| awning | object | -2.785 |
| saddle | part | -2.777 |
| building | object | -2.458 |
| slot machine | object | -2.373 |
| balcony | part | -2.357 |
| windowpane | object | -2.337 |

**decomposition**

| concept | category | value |
|---|---|---|
| floor | object | 0.3 |
| casino-indoor-s | scene | -0.2 |
| counter | object | 0.149 |
| conference_room-s | scene | -0.161 |
| building | object | -0.137 |

**leftover**: 0.803

**examples**

| concept | category | value |
|---|---|---|
| floor | object | 0.031 |
| road | object | 0.017 |
| sidewalk | object | 0.016 |
| grey-c | color | 0.014 |
| head | part | 0.014 |

# Read: child n385

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| chandelier | object | 4.124 |
| vase | object | 3.739 |
| pitted | texture | 3.617 |
| bubbly | texture | 3.571 |
| boat | object | 3.498 |
| fan | object | 3.41 |
| work surface | object | 3.404 |
| sink | object | 3.363 |

**down**

| concept | category | value |
|---|---|---|
| striped | texture | -4.416 |
| chequered | texture | -3.732 |
| elevator / elevator door | object | -2.802 |
| banded | texture | -2.742 |
| silver screen | object | -2.556 |
| meshed | texture | -2.485 |
| braided | texture | -2.381 |
| pane | part | -2.362 |

**decomposition**

| concept | category | value |
|---|---|---|
| boat | object | 0.204 |
| striped | texture | -0.158 |
| work surface | object | 0.189 |
| chequered | texture | -0.167 |
| windowpane | object | -0.164 |

**leftover**: 0.824

**examples**

| concept | category | value |
|---|---|---|
| table | object | 0.013 |
| boat | object | 0.013 |
| bathroom-s | scene | 0.013 |
| wheel | part | 0.012 |
| kitchen-s | scene | 0.011 |

# Write: child n385

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| bar | object | 0.233 |
| ring | object | 0.182 |
| bar-s | scene | 0.165 |
| stage | object | 0.159 |
| case | object | 0.155 |
| grandstand | object | 0.151 |
| boat | object | 0.151 |
| dining_room-s | scene | 0.144 |

**down**

| concept | category | value |
|---|---|---|
| rock | object | -0.169 |
| matted | texture | -0.155 |
| forest-broadleaf-s | scene | -0.155 |
| paw | part | -0.153 |
| fabric | material | -0.153 |
| creek-s | scene | -0.15 |
| forest-needleleaf-s | scene | -0.149 |
| waterfall | object | -0.148 |

**decomposition**

| concept | category | value |
|---|---|---|
| bar | object | 0.344 |
| ottoman | object | -0.23 |
| matted | texture | -0.207 |
| file cabinet | object | -0.168 |
| market-outdoor-s | scene | -0.109 |

**leftover**: 0.859

**examples**

| concept | category | value |
|---|---|---|
| dining_room-s | scene | 0.014 |
| boat | object | 0.012 |
| bicycle | object | 0.011 |
| black-c | color | 0.011 |
| wheel | part | 0.011 |

# Read: child n385 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| pitted | texture | 3.75 |
| boat | object | 3.447 |
| chandelier | object | 3.417 |
| bubbly | texture | 3.411 |
| vase | object | 3.332 |
| work surface | object | 3.274 |
| porous | texture | 3.208 |
| wineglass | object | 3.061 |

**down**

| concept | category | value |
|---|---|---|
| striped | texture | -4.408 |
| chequered | texture | -3.789 |
| elevator / elevator door | object | -2.829 |
| banded | texture | -2.706 |
| silver screen | object | -2.667 |
| meshed | texture | -2.466 |
| pane | part | -2.389 |
| braided | texture | -2.358 |

**decomposition**

| concept | category | value |
|---|---|---|
| boat | object | 0.209 |
| pitted | texture | 0.14 |
| work surface | object | 0.181 |
| chequered | texture | -0.162 |
| windowpane | object | -0.155 |

**leftover**: 0.833

**examples**

| concept | category | value |
|---|---|---|
| boat | object | 0.015 |
| table | object | 0.014 |
| bathroom-s | scene | 0.012 |
| brown-c | color | 0.011 |
| sink | object | 0.011 |

# Read: child n297

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| stained | texture | 4.199 |
| blotchy | texture | 4.169 |
| smeared | texture | 4.134 |
| marbled | texture | 4.103 |
| poster | object | 3.184 |
| plant | object | 3.165 |
| flower | object | 3.103 |
| bush | object | 2.784 |

**down**

| concept | category | value |
|---|---|---|
| perforated | texture | -3.771 |
| meshed | texture | -3.173 |
| grid | texture | -2.997 |
| window_seat-s | scene | -2.96 |
| roof | part | -2.931 |
| stairway | object | -2.888 |
| dinette-vehicle-s | scene | -2.793 |
| woven | texture | -2.79 |

**decomposition**

| concept | category | value |
|---|---|---|
| smeared | texture | 0.249 |
| plant | object | 0.182 |
| poster | object | 0.179 |
| dinette-vehicle-s | scene | -0.16 |
| person | object | -0.154 |

**leftover**: 0.809

**examples**

| concept | category | value |
|---|---|---|
| wheel | part | 0.02 |
| painting | object | 0.018 |
| bedroom-s | scene | 0.011 |
| car | object | 0.011 |
| plant | object | 0.011 |

# Write: child n297

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| paisley | texture | 0.304 |
| poster | object | 0.288 |
| trade name | object | 0.276 |
| paper | material | 0.26 |
| tapestry | object | 0.249 |
| signboard | object | 0.211 |
| painting | object | 0.207 |
| swirly | texture | 0.189 |

**down**

| concept | category | value |
|---|---|---|
| grooved | texture | -0.192 |
| basement-s | scene | -0.166 |
| crosshatched | texture | -0.144 |
| flecked | texture | -0.143 |
| woven | texture | -0.143 |
| black-c | color | -0.142 |
| meshed | texture | -0.137 |
| artists_loft-s | scene | -0.136 |

**decomposition**

| concept | category | value |
|---|---|---|
| paisley | texture | 0.201 |
| poster | object | 0.258 |
| ashcan | object | -0.228 |
| plastic-clear | material | 0.179 |
| knitted | texture | -0.147 |

**leftover**: 0.794

**examples**

| concept | category | value |
|---|---|---|
| painting | object | 0.026 |
| zigzagged | texture | 0.022 |
| paisley | texture | 0.017 |
| yellow-c | color | 0.014 |
| bedroom-s | scene | 0.01 |

# Read: child n297 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| blotchy | texture | 4.328 |
| marbled | texture | 4.32 |
| stained | texture | 4.311 |
| smeared | texture | 4.266 |
| plant | object | 3.157 |
| poster | object | 3.064 |
| bush | object | 2.984 |
| forest-broadleaf-s | scene | 2.906 |

**down**

| concept | category | value |
|---|---|---|
| perforated | texture | -3.899 |
| meshed | texture | -3.204 |
| grid | texture | -3.05 |
| window_seat-s | scene | -2.99 |
| roof | part | -2.907 |
| stairway | object | -2.892 |
| poolroom-home-s | scene | -2.888 |
| baggage_claim-s | scene | -2.879 |

**decomposition**

| concept | category | value |
|---|---|---|
| smeared | texture | 0.263 |
| torso | part | -0.162 |
| bush | object | 0.185 |
| poster | object | 0.179 |
| dinette-vehicle-s | scene | -0.158 |

**leftover**: 0.803

**examples**

| concept | category | value |
|---|---|---|
| painting | object | 0.02 |
| wheel | part | 0.012 |
| plant | object | 0.012 |
| bedroom-s | scene | 0.011 |
| brown-c | color | 0.01 |

# Read: child n464

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| interlaced | texture | 3.74 |
| paisley | texture | 3.045 |
| food | material | 2.861 |
| eye | part | 2.718 |
| swirly | texture | 2.617 |
| ceramic | material | 2.567 |
| plate | object | 2.489 |
| creek-s | scene | 2.485 |

**down**

| concept | category | value |
|---|---|---|
| dishwasher | object | -4.158 |
| chequered | texture | -4.155 |
| tvmonitor | object | -4.074 |
| screen | part | -3.894 |
| cage / bird cage | object | -3.773 |
| artists_loft-s | scene | -3.636 |
| train | object | -3.268 |
| monitor | part | -3.261 |

**decomposition**

| concept | category | value |
|---|---|---|
| tvmonitor | object | -0.188 |
| tile | material | -0.266 |
| black-c | color | -0.197 |
| windowpane | object | -0.192 |
| dishwasher | object | -0.167 |

**leftover**: 0.77

**examples**

| concept | category | value |
|---|---|---|
| head | part | 0.036 |
| interlaced | texture | 0.034 |
| cat | object | 0.028 |
| wheel | part | 0.026 |
| dog | object | 0.023 |

# Write: child n464

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.305 |
| interlaced | texture | 0.278 |
| spiralled | texture | 0.251 |
| house | object | 0.234 |
| tower | object | 0.215 |
| wardrobe | object | 0.21 |
| balcony | part | 0.21 |
| door | part | 0.207 |

**down**

| concept | category | value |
|---|---|---|
| creek-s | scene | -0.279 |
| pitted | texture | -0.256 |
| forest-broadleaf-s | scene | -0.247 |
| blotchy | texture | -0.245 |
| porous | texture | -0.243 |
| field-wild-s | scene | -0.237 |
| bush | object | -0.219 |
| flecked | texture | -0.218 |

**decomposition**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.253 |
| spiralled | texture | 0.28 |
| screen door | object | 0.229 |
| plaza-s | scene | 0.143 |
| airplane | object | -0.096 |

**leftover**: 0.811

**examples**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.064 |
| building | object | 0.023 |
| interlaced | texture | 0.023 |
| orange-c | color | 0.015 |
| house | object | 0.013 |

# Read: child n464 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| interlaced | texture | 3.679 |
| paisley | texture | 2.929 |
| creek-s | scene | 2.578 |
| food | material | 2.533 |
| eye | part | 2.521 |
| swirly | texture | 2.495 |
| castle-s | scene | 2.478 |
| ceramic | material | 2.41 |

**down**

| concept | category | value |
|---|---|---|
| dishwasher | object | -4.182 |
| chequered | texture | -4.148 |
| tvmonitor | object | -4.106 |
| screen | part | -3.959 |
| cage / bird cage | object | -3.61 |
| artists_loft-s | scene | -3.553 |
| train | object | -3.388 |
| monitor | part | -3.316 |

**decomposition**

| concept | category | value |
|---|---|---|
| tvmonitor | object | -0.202 |
| tile | material | -0.266 |
| train | object | -0.153 |
| black-c | color | -0.177 |
| windowpane | object | -0.183 |

**leftover**: 0.778

**examples**

| concept | category | value |
|---|---|---|
| interlaced | texture | 0.036 |
| head | part | 0.029 |
| paisley | texture | 0.023 |
| cat | object | 0.022 |
| dog | object | 0.021 |

# Read: child n237

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| lacelike | texture | 4.597 |
| sprinkled | texture | 4.243 |
| amusement_park-s | scene | 3.789 |
| bubbly | texture | 3.594 |
| landing_deck-s | scene | 3.52 |
| flower | object | 3.478 |
| mountain_snowy-s | scene | 3.206 |
| paisley | texture | 3.198 |

**down**

| concept | category | value |
|---|---|---|
| cradle | object | -4.41 |
| radiator | object | -4.322 |
| kitchen island | object | -3.574 |
| bannister | object | -3.416 |
| pleated | texture | -3.393 |
| window_seat-s | scene | -3.304 |
| back pillow | part | -3.269 |
| headboard | part | -3.266 |

**decomposition**

| concept | category | value |
|---|---|---|
| cabinet | object | -0.243 |
| lacelike | texture | 0.186 |
| sprinkled | texture | 0.167 |
| armchair | object | -0.193 |
| house | object | -0.185 |

**leftover**: 0.795

**examples**

| concept | category | value |
|---|---|---|
| dotted | texture | 0.061 |
| studded | texture | 0.032 |
| lacelike | texture | 0.024 |
| perforated | texture | 0.02 |
| dog | object | 0.016 |

# Write: child n237

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| polka-dotted | texture | 0.265 |
| dotted | texture | 0.259 |
| studded | texture | 0.258 |
| swirly | texture | 0.221 |
| paisley | texture | 0.212 |
| kitchen island | object | 0.159 |
| perforated | texture | 0.159 |
| zigzagged | texture | 0.154 |

**down**

| concept | category | value |
|---|---|---|
| pitted | texture | -0.22 |
| cracked | texture | -0.22 |
| blotchy | texture | -0.204 |
| porous | texture | -0.194 |
| stained | texture | -0.188 |
| flecked | texture | -0.182 |
| marbled | texture | -0.174 |
| crystalline | texture | -0.167 |

**decomposition**

| concept | category | value |
|---|---|---|
| polka-dotted | texture | 0.271 |
| wood | material | 0.155 |
| paisley | texture | 0.232 |
| guitar | object | -0.248 |
| seat cushion | part | 0.19 |

**leftover**: 0.806

**examples**

| concept | category | value |
|---|---|---|
| dotted | texture | 0.063 |
| zigzagged | texture | 0.044 |
| studded | texture | 0.039 |
| polka-dotted | texture | 0.036 |
| perforated | texture | 0.025 |

# Read: child n237 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| lacelike | texture | 4.583 |
| sprinkled | texture | 4.132 |
| amusement_park-s | scene | 3.793 |
| bubbly | texture | 3.434 |
| mountain_snowy-s | scene | 3.384 |
| landing_deck-s | scene | 3.359 |
| dotted | texture | 3.15 |
| highway-s | scene | 3.121 |

**down**

| concept | category | value |
|---|---|---|
| cradle | object | -4.527 |
| radiator | object | -4.341 |
| kitchen island | object | -3.775 |
| bannister | object | -3.511 |
| back pillow | part | -3.431 |
| pleated | texture | -3.417 |
| footboard | part | -3.4 |
| headboard | part | -3.398 |

**decomposition**

| concept | category | value |
|---|---|---|
| cabinet | object | -0.251 |
| armchair | object | -0.206 |
| house | object | -0.185 |
| lacelike | texture | 0.186 |
| sprinkled | texture | 0.162 |

**leftover**: 0.787

**examples**

| concept | category | value |
|---|---|---|
| dotted | texture | 0.079 |
| studded | texture | 0.036 |
| lacelike | texture | 0.03 |
| perforated | texture | 0.026 |
| paisley | texture | 0.017 |

# Read: child n108

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| wheel | part | 3.255 |
| motorbike | object | 2.694 |
| bicycle | object | 2.523 |
| headlight | part | 2.13 |
| chandelier | object | 2.108 |
| barrel | object | 1.949 |
| car | object | 1.926 |
| cockpit-s | scene | 1.841 |

**down**

| concept | category | value |
|---|---|---|
| chequered | texture | -2.183 |
| dishwasher | object | -2.157 |
| youth_hostel-s | scene | -2.15 |
| pink-c | color | -2.038 |
| skin | material | -2.012 |
| baggage_claim-s | scene | -1.997 |
| elevator / elevator door | object | -1.97 |
| dotted | texture | -1.921 |

**decomposition**

| concept | category | value |
|---|---|---|
| wheel | part | 0.18 |
| floor | object | -0.143 |
| stained | texture | 0.119 |
| refrigerator | object | -0.126 |
| armchair | object | 0.125 |

**leftover**: 0.913

**examples**

| concept | category | value |
|---|---|---|
| car | object | 0.038 |
| wheel | part | 0.031 |
| motorbike | object | 0.017 |
| black-c | color | 0.015 |
| bicycle | object | 0.014 |

# Write: child n108

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| car | object | 0.23 |
| wheel | part | 0.227 |
| truck | object | 0.221 |
| motorbike | object | 0.198 |
| van | object | 0.184 |
| washer | object | 0.178 |
| bus | object | 0.177 |
| spiralled | texture | 0.175 |

**down**

| concept | category | value |
|---|---|---|
| flecked | texture | -0.207 |
| ocean-s | scene | -0.2 |
| forest-broadleaf-s | scene | -0.163 |
| creek-s | scene | -0.153 |
| crosshatched | texture | -0.151 |
| ice rink | object | -0.15 |
| coast-s | scene | -0.149 |
| highway-s | scene | -0.146 |

**decomposition**

| concept | category | value |
|---|---|---|
| car | object | 0.441 |
| flecked | texture | -0.233 |
| body | part | -0.268 |
| alley-s | scene | 0.177 |
| subway_interior-s | scene | -0.162 |

**leftover**: 0.836

**examples**

| concept | category | value |
|---|---|---|
| car | object | 0.086 |
| zigzagged | texture | 0.032 |
| wheel | part | 0.031 |
| dog | object | 0.028 |
| motorbike | object | 0.025 |

# Read: child n108 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| wheel | part | 2.593 |
| motorbike | object | 2.297 |
| bicycle | object | 1.998 |
| barrel | object | 1.899 |
| cockpit-s | scene | 1.798 |
| headlight | part | 1.764 |
| matted | texture | 1.665 |
| potholed | texture | 1.654 |

**down**

| concept | category | value |
|---|---|---|
| chequered | texture | -2.158 |
| dishwasher | object | -2.151 |
| youth_hostel-s | scene | -2.126 |
| pink-c | color | -2.074 |
| nose | part | -2.033 |
| skin | material | -2.025 |
| baggage_claim-s | scene | -2.025 |
| dotted | texture | -2.025 |

**decomposition**

| concept | category | value |
|---|---|---|
| motorbike | object | 0.16 |
| dishwasher | object | -0.112 |
| floor | object | -0.122 |
| stained | texture | 0.113 |
| bottle | object | -0.108 |

**leftover**: 0.929

**examples**

| concept | category | value |
|---|---|---|
| car | object | 0.032 |
| wheel | part | 0.025 |
| black-c | color | 0.015 |
| motorbike | object | 0.015 |
| bicycle | object | 0.011 |

# Read: child n510

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| bubbly | texture | 4.876 |
| paisley | texture | 4.801 |
| bumpy | texture | 4.414 |
| plastic-clear | material | 3.131 |
| embankment | object | 3.123 |
| flower | object | 3.011 |
| food | material | 3.009 |
| market-outdoor-s | scene | 2.83 |

**down**

| concept | category | value |
|---|---|---|
| plaza-s | scene | -4.008 |
| apartment_building-outdoor-s | scene | -3.955 |
| striped | texture | -3.628 |
| chequered | texture | -3.404 |
| grid | texture | -2.904 |
| arrival_gate-outdoor-s | scene | -2.897 |
| crosshatched | texture | -2.848 |
| shutter | part | -2.846 |

**decomposition**

| concept | category | value |
|---|---|---|
| sky | object | -0.179 |
| bubbly | texture | 0.153 |
| paisley | texture | 0.269 |
| embankment | object | 0.203 |
| bumpy | texture | 0.195 |

**leftover**: 0.795

**examples**

| concept | category | value |
|---|---|---|
| paisley | texture | 0.05 |
| swirly | texture | 0.024 |
| studded | texture | 0.024 |
| head | part | 0.023 |
| interlaced | texture | 0.019 |

# Write: child n510

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| studded | texture | 0.252 |
| paisley | texture | 0.234 |
| swirly | texture | 0.194 |
| staircase-s | scene | 0.15 |
| polka-dotted | texture | 0.145 |
| dotted | texture | 0.145 |
| windmill | object | 0.134 |
| balcony-interior-s | scene | 0.13 |

**down**

| concept | category | value |
|---|---|---|
| matted | texture | -0.258 |
| fibrous | texture | -0.227 |
| pitted | texture | -0.221 |
| marbled | texture | -0.218 |
| porous | texture | -0.218 |
| blotchy | texture | -0.198 |
| potholed | texture | -0.185 |
| stratified | texture | -0.178 |

**decomposition**

| concept | category | value |
|---|---|---|
| matted | texture | -0.274 |
| paisley | texture | 0.317 |
| trade name | object | -0.24 |
| wing | part | 0.19 |
| barrel | object | -0.156 |

**leftover**: 0.81

**examples**

| concept | category | value |
|---|---|---|
| studded | texture | 0.041 |
| paisley | texture | 0.04 |
| swirly | texture | 0.031 |
| interlaced | texture | 0.023 |
| bumpy | texture | 0.018 |

# Read: child n510 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| paisley | texture | 4.718 |
| bubbly | texture | 4.716 |
| bumpy | texture | 4.386 |
| embankment | object | 3.273 |
| plastic-clear | material | 3.046 |
| rock | object | 2.869 |
| bedclothes | object | 2.853 |
| green-c | color | 2.735 |

**down**

| concept | category | value |
|---|---|---|
| plaza-s | scene | -4.045 |
| apartment_building-outdoor-s | scene | -3.843 |
| striped | texture | -3.553 |
| chequered | texture | -3.403 |
| arrival_gate-outdoor-s | scene | -2.992 |
| grid | texture | -2.887 |
| crosshatched | texture | -2.798 |
| shutter | part | -2.769 |

**decomposition**

| concept | category | value |
|---|---|---|
| paisley | texture | 0.287 |
| bumpy | texture | 0.254 |
| embankment | object | 0.21 |
| plaza-s | scene | -0.167 |
| fabric | material | 0.142 |

**leftover**: 0.792

**examples**

| concept | category | value |
|---|---|---|
| paisley | texture | 0.055 |
| swirly | texture | 0.026 |
| studded | texture | 0.024 |
| head | part | 0.021 |
| interlaced | texture | 0.02 |

# Read: child n275

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| chandelier | object | 5.225 |
| fan | object | 4.31 |
| flower | object | 3.197 |
| zigzagged | texture | 2.889 |
| lacelike | texture | 2.625 |
| palm | object | 2.616 |
| skylight | object | 2.516 |
| light | object | 2.213 |

**down**

| concept | category | value |
|---|---|---|
| spiralled | texture | -4.06 |
| eye | part | -3.697 |
| bus | object | -3.524 |
| washer | object | -3.518 |
| car | object | -3.332 |
| van | object | -3.268 |
| headlight | part | -3.103 |
| swirly | texture | -2.851 |

**decomposition**

| concept | category | value |
|---|---|---|
| bus | object | -0.218 |
| chandelier | object | 0.199 |
| spiralled | texture | -0.176 |
| zigzagged | texture | 0.143 |
| ceiling | object | 0.117 |

**leftover**: 0.833

**examples**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.038 |
| interlaced | texture | 0.018 |
| ceiling | object | 0.018 |
| grid | texture | 0.014 |
| person | object | 0.014 |

# Write: child n275

**kind**: write

**metric**: raw

**score**: cosine

**up**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.451 |
| chequered | texture | 0.35 |
| interlaced | texture | 0.269 |
| dome | part | 0.206 |
| studded | texture | 0.205 |
| grid | texture | 0.202 |
| chandelier | object | 0.197 |
| honeycombed | texture | 0.196 |

**down**

| concept | category | value |
|---|---|---|
| potholed | texture | -0.307 |
| marbled | texture | -0.304 |
| pitted | texture | -0.298 |
| blotchy | texture | -0.298 |
| fibrous | texture | -0.294 |
| matted | texture | -0.285 |
| porous | texture | -0.276 |
| flecked | texture | -0.272 |

**decomposition**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.351 |
| chandelier | object | 0.226 |
| classroom-s | scene | -0.199 |
| chequered | texture | 0.263 |
| apartment_building-outdoor-s | scene | -0.144 |

**leftover**: 0.695

**examples**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.095 |
| chequered | texture | 0.083 |
| interlaced | texture | 0.053 |
| grid | texture | 0.037 |
| studded | texture | 0.027 |

# Read: child n275 condition

**kind**: read

**metric**: whitened

**score**: z vs random reads

**up**

| concept | category | value |
|---|---|---|
| chandelier | object | 4.615 |
| fan | object | 3.621 |
| zigzagged | texture | 2.956 |
| flower | object | 2.785 |
| palm | object | 2.552 |
| lacelike | texture | 2.518 |
| skylight | object | 2.518 |
| screen door | object | 2.122 |

**down**

| concept | category | value |
|---|---|---|
| spiralled | texture | -4.143 |
| eye | part | -3.917 |
| washer | object | -3.68 |
| bus | object | -3.664 |
| car | object | -3.641 |
| headlight | part | -3.482 |
| van | object | -3.441 |
| swirly | texture | -3.023 |

**decomposition**

| concept | category | value |
|---|---|---|
| car | object | -0.222 |
| chandelier | object | 0.195 |
| spiralled | texture | -0.16 |
| zigzagged | texture | 0.147 |
| building | object | -0.136 |

**leftover**: 0.838

**examples**

| concept | category | value |
|---|---|---|
| zigzagged | texture | 0.041 |
| interlaced | texture | 0.022 |
| ceiling | object | 0.017 |
| grid | texture | 0.017 |
| person | object | 0.014 |
