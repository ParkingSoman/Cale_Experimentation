# Family judging (L3_input)

Each case has one PARENT unit and two sets of other units (X and Y). Exactly one set holds units that the
parent was found to cover. Pick the set for which the parent is best described as a broader, coarser
category that includes the units in the set. Give a confidence 1 (guess) to 5 (sure).


## F1

PARENT: Fires on contraction and possessive endings after an apostrophe (the s in 's, re in 're, t in n't) and on the second = of wiki heading markers.
Set X:
  - Fires on the middle or final piece of a split proper name or rare word (Shell-back, Sh-ima, touch-down), and sometimes on 'a' after 'was'.
  - Fires on the unit or count noun after a number or quantity (rounds, long tons, half, jump) and on suffix pieces of long abstract words (-idity, -ative); also 'spell/points/stages' after a modifier.
  - Fires on mid-word fragments of rare names and on 'Ten' after 'Big', and on the first digit of a chart number after 'number'.
  - Fires on the next piece of a split first name or word right after a preceding name part (Jon Dol-, Jos-, Justin, touch-down) and on small function words like 'in' or 'around' after a verb.
Set Y:
  - Fires on the word part right after an apostrophe in contractions and possessives ('re, 've, 's, 't), plus the h in km / h.
  - Fires on the final = of a wiki section heading marker, and on mid-word fragments of rare names and long words (Rami, dega-ussing, Mus-well).
  - Fires on the 's or 're suffix after an apostrophe, on the second byte of broken CJK characters, and on the word 'room' after 'engine'.
  - Mixed: fires on the second = of section heading markers, on the word 'Factor' after 'The X', and on fragments of rare names (Petos-key, Cris-p).

## F2

PARENT: Fires on the hyphen or @ symbol in the @-@ hyphen pattern (scholar @-@, little @-@) and on the second = of heading markers.
Set X:
  - Fires on connective function words that start an adverbial or link phrase (more broadly, into, for, as a result, most, In contrast, drawn from).
  - Fires on the last piece of a split proper name, mostly a surname or place (Petos-key, Tiff-any, Pres-ley, Sh-ane, S-ich).
  - Fires on the later pieces of split rare words and names (Mercen-aries, Don-key, rejo-ined, Ne-wn, Kim Ne-ely).
  - Fires on the last token of a name or place phrase (Jocely-n, Simon Cow-ell, Arthur Hill, Bonin Islands, Chinese coast) and on broken CJK bytes.
Set Y:
  - Fires on the @ symbol in @-@ or @.@ number/hyphen patterns, and on the first digit of a caliber number after 'calibre' (45-calibre [[3]]).
  - Fires on the second @ of the @-@ hyphen pattern and on heading = markers, plus on middle fragments of rare names (Indoch-ina, Mei Yao-c).
  - Mixed: fires on the second @ of @-@ after a word, and on mid-word pieces of names (Ky-ush-u, Saihok-ush) and on adverbs like 'temporary' or 'immediately'.
  - Fires on the @ in decimal and hyphen patterns (3 @.@ 429) and on the second = of heading markers such as '= = Charts ='.

## F3

PARENT: Fires on the hyphen after @ in @-@, on the second byte of broken CJK characters, on heading = markers, and on mid-word fragments (Pet-rol, gun-nery).
Set X:
  - Mixed: fires on word-final fragments (Cred-its, lab-oured, Val-erie, Bast-et) and on 'possible' after 'as ... as'.
  - Fires on the last digit of a year, mostly the third or fourth digit of 3-digit and 4-digit years (75[[5]], 191[[3]]).
  - Fires on the word 'been' or 'a' and similar helper words in the middle of a clause, and on some name endings (Cow-ell, H-ie).
  - Mixed: fires on mid-word fragments of rare names and on adjectives or adverbs after 'was/very/too' (thin, late, supreme), and on digits in numbers.
Set Y:
  - Mixed: fires on the hyphen of @-@ patterns, on name fragments (K-ier, Baid-), and on closing parentheses after a date or measurement.
  - Mixed: fires on the second = of heading markers, on suffix fragments (thatched, anthe-mic, a-irst), and on 'day' in 'late in the day'.
  - Fires on the second = of heading markers and the hyphen of @-@ patterns, on broken CJK bytes, and on the noun 'capacity' or 'hang(ar)'.
  - Fires on the hyphen in @-@ after a word and on middle or end fragments of rare names and words (Rid-ley, Yac-oub, solips-ism).

## F4

PARENT: Fires on the third digit of a 4-digit year after '19' (19[[1]], 19[[3]]) and on the unit 'ft' after a number.
Set X:
  - Mixed: fires on word pieces in rare names (Her-ons, Mark St-ille, H-orthy) and on 'previously' or 'follows' after 'had' or 'as'.
  - Mixed: fires on the unit or magnitude word after a number (metres, in, inches, million), on 'part' after 'In/As', and on the split name pieces Pred-yn and Anub-is.
  - Mixed: fires on mid-word fragments (esc-orts, epit-he, Ed-fu, Battles-hip) and on words like 'top', 'dive', 'room' after an adjective.
  - Mixed: fires on 'Factor' after 'The X', on 'more' after 'or', 'of' after 'a pair', and on end pieces of rare names.
Set Y:
  - Mixed: fires on the @ in numbers like 7 @.@ 62 (25 [[@]]), on 'more' or 'life' in loose phrases, and on word-end fragments.
  - Fires on the unit word after a number in a measurement (metres, ft, feet, inches, m), mostly length.
  - Fires on the @ in numbers like 53 @-@ and 7 @,@, and on the stem 'arm' (armour, armament) and the digit 9 in type numbers.
  - Fires on verbs and nouns that continue a phrase after a subject (focusing, intention, reassess, themes), and on the @ in big numbers.

## F5

PARENT: Fires on closing quote marks and apostrophes (a quotation mark or single quote right after a quoted word) and on the 's in possessives; also the second @ and 'ft'.
Set X:
  - Mixed: fires on middle fragments of rare names and words (Arsen-al, Er-z, Merc-en) and on words like 'around', 'or', 'tubes' in mid-phrase.
  - Fires on the second word of a multi-word proper name or noun phrase (green screen, Rockefeller Center, Rolling Stone, South Pacific, Crown Prince, Black Sabbath).
  - Mixed: fires on mid-word fragments of rare names and words (ha-iku, RI-AA, Repulse, Har-p, break-out), and on broken CJK bytes.
  - Fires on a word that introduces a name or description after a noun (named, called, titled, designated, consisted of, example of).
Set Y:
  - Fires on the apostrophe or curly quote right after a word (Fu ', Teddy ', don ’) and on 'War' in 'World War', plus broken CJK bytes.
  - Fires on a closing quotation mark or apostrophe after a quoted word, and on 's in possessives and plural -ars/-ima endings.
  - Fires on a quote mark or apostrophe that closes or follows a quoted word (Kiss You ", Prince ', don ’), and on word-middle pieces like -ast, -od.
  - Mixed: fires on the hyphen-@ in 'Ise @-@' headings, on the final piece of rare names (Z-aza, G-id, Nunn), and on shp.

## F6

PARENT: Fires on the unit after a number in a measurement (ft, mm, cm, km, metres), especially length or gun caliber.
Set X:
  - Mixed: fires on later pieces of split rare names and words (Holy-o, Shuns-ai, Shū, Gans-u, surf) and on broken CJK bytes.
  - Fires on middle or later pieces of split foreign or rare names (Z-aza, Ch-ou, Yaoc-hen, Giles, Indoch-ina).
  - Mixed: fires on 'Fu' after 'Du', on 'which' after 'each of', and on mid-name fragments (Huff-ington, Ful-ham, Xuan, ph-ara-oh).
  - Mixed: fires on 'Direction' after 'One', on 'long' in 'a long ...', and on mid-word fragments (G-id, un-happiness, arm-ament).
Set Y:
  - Fires on the unit after a number in a measurement or speed (km, mph, ft, m, mi, metres); also on -ions in 'processions'.
  - Fires on the unit word after a number for distance, range or count (mi, miles, km, ft, nautical, views, hours).
  - Mixed: fires on commas in lists after a noun, on broken CJK bytes, and on mid-word pieces (Hot-ch, Tong-gu, Dao-ist).
  - Fires on the unit after a number in a measurement (feet, metres, ft), and on mid-word pieces (touch-down, D-ool, Z-ay).

## F7

PARENT: Fires on the second @ in @-@ patterns (anti @-[[@]]) and on pieces of broken characters or rare words (casem-, truisms).
Set X:
  - Mixed: fires on the second word of a title or place name (Hall of Fame, Third Fleet, New York City, second studio, ninth studio) and on broken CJK bytes.
  - Mixed: fires on later pieces of rare names (Ned-ler, Lemie-ux, Dread-n), and on function words like 'to' after 'unable/decided'.
  - Fires on a trailing piece of a technical noun split into tokens (bulk-heads, catap-ult, war-head, mumm-ified, sph-inx).
  - Mixed: fires on 'versions'/'version' after a modifier (US and UK versions, live version), and on later pieces of names (Bai Ju, Mein, Hotchk-iss, Ne-ith).
Set Y:
  - Fires on the @ and first digit in the middle of numbers with comma or decimal markup (2 @,[[@]], ( [[4]]).
  - Mixed: fires on the @ in numbers and hyphens (3 @.[[@]], all [[ @]]), on fragments like epit-het, ora-cles, and on 'all'.
  - Mixed: fires on the @ in hyphen patterns and on assorted mid-sentence words and number pieces (form, now, 193[[0]], 10[[th]]).
  - Mixed: fires on mid-name fragments (Ash-ik-aga, D-read, Hash-ira), on the @ in numbers (20 @,[[@]]), and on 'same'.

## F8

PARENT: Fires on the @ or hyphen symbol in @-@ patterns (anti @-@, scholar @-), and on the second = of heading markers.
Set X:
Set Y:

## F9

PARENT: Fires on the last = in a heading marker, on the @ in decimals and hyphens, and on broken CJK or rare names (An L-ush-an).
Set X:
Set Y:

## F10

PARENT: Mixed: fires on word-end pieces (Black-burn, touch-down, stra-ke, Tru-k), on 'of' in 'make use of', and on 'way' in 'make their way'.
Set X:
Set Y:
