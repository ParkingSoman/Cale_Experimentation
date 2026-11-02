# Parent vs children: is the parent broader?

For each case: one PARENT unit and the units it covers (CHILDREN). Rate the relation.


## C001

PARENT: Fires on the "(" or first digit that opens a unit conversion after miles, tons, inches or mm.
CHILDREN:
  - Fires on the first digit or space after "(" in conversions following tons, knots or km.
  - Fires on the first digit right after "(" in a mm or cm armour and gun size conversion.
  - Fires on "(" after mm or tons, digits after it, and "=" ending wiki headings.
  - Fires on "(" after tons or tonnes, the first digit after it, and "more" or "but" in comparisons.

## C002

PARENT: Fires on a subject, adverb or auxiliary before a verb in chart narrative ('has [been]', '[was]', '[It]', '[He]', '[They]', 'ultimately', 'fourth'), predicting a verb.
CHILDREN:
  - Fires on ordinals and auxiliaries in chart and album facts ('the group's [sixth]', 'its [third] and [fourth]', 'has [been]', 'was', 'to [be]'), predicting a noun or participle.
  - Fires on the word just before a verb in chart or career facts ('the track [also]', 'the song [had]', 'neither air group [ever]', 'It', 'He'), predicting a verb.
  - Fires on an adverb or subject before the verb ('previously', 'briefly', 'who', 'He', 'again'), predicting a verb.
  - Fires on a subject right before a verb in biography ('He', 'he', 'and', 'always', 'have', 'all'), also on the name piece Rif-enburg.

## C003

PARENT: Fires on the second @ in @-@ patterns (anti @-[[@]]) and on pieces of broken characters or rare words (casem-, truisms).
CHILDREN:
  - Fires on the @ and first digit in the middle of numbers with comma or decimal markup (2 @,[[@]], ( [[4]]).
  - Mixed: fires on the @ in numbers and hyphens (3 @.[[@]], all [[ @]]), on fragments like epit-het, ora-cles, and on 'all'.
  - Mixed: fires on the @ in hyphen patterns and on assorted mid-sentence words and number pieces (form, now, 193[[0]], 10[[th]]).
  - Mixed: fires on mid-name fragments (Ash-ik-aga, D-read, Hash-ira), on the @ in numbers (20 @,[[@]]), and on 'same'.

## C004

PARENT: Fires on varied tokens near flight-deck and dimension text ('flight [deck]', 'had', 'by', digit 9, '(', 's', 'its', 'granted'); mixed.
CHILDREN:
  - Fires on words in actor career sentences ('received [a] [favorable] review', 'landed [a]', '[was]'), but also on year digits and name pieces; loosely a career-narration unit.
  - Fires on a digit inside a measurement (95 ft, 104 ft, 9,440, 219 m, 414 ft, 71 ft), predicting another digit.

## C005

PARENT: Fires on a subject or adverb right before the verb in biography and chart text ('had [previously]', '[he]', '[and]', '[He]', '[also]', 'He [played]', 'has [since]'), predicting a verb.
CHILDREN:
  - Fires on modifiers in 'guest-starring role' and 'recurring role' phrases ('a [guest]', '[starring]', '[recurring]', 'had a'), and on 'to', 'who' in the Egyptian text; mostly an actor-credit unit.
  - Fires on assorted tokens: 'television', Rid-ley, 'the', 'UK', 'two', digits, adjectives like 'developed', 'cohesive', 'elaborate'; mixed.
  - Fires on ordinals and articles in chart or career facts ('their [second] studio album', 'the group's [sixth]', 'had [a] role'), predicting the next noun of the sequence.
  - Fires on 'the television', 'which', 'play', 'theatre' and the fragment ley in 'Ridley'; mixed.

## C006

PARENT: Fires on an apostrophe or curly quote that marks a possessive or contraction (group', don', team', father'), and on the second '=' of a section heading marker. It marks where an 's or 't is about to follow.
CHILDREN:
  - Fires on an apostrophe or closing quote after a noun (father', ship', gods', brother') just before the possessive 's, and on dashes or hyphens in ranges. A mostly possessive apostrophe pattern with some strays.
  - Fires on an apostrophe before 's (group', Rubin', brother'), and on the second '=' of a heading marker, the '@' of a split hyphen (non @-@), or a space inside a parenthesis. A mix of punctuation tokens.
  - Fires on an apostrophe before a possessive 's, on an ordinal suffix 'th' after a number (58th, 24th), and on commas inside dates. A mix of possessive apostrophes and number/date suffix tokens.
  - Fires on an apostrophe (straight or curly) just before 's, 't or 'll in a possessive or contraction (protagonist', It', don'), and on some word-pieces such as 'in' or 'om' of 'becoming'.

## C007

PARENT: Fires on the first or middle digit of a number or year (1, 2, 0, 9), or on the space before a date number, predicting more digits.
CHILDREN:
  - Fires on digits inside dates, years and times (7-7-0, 1-9, 2-6, 1-3), predicting more digits.
  - Fires on the first digit of a year or number and on approximate-time words ('around', 'late', 'between'), predicting a date.
  - Fires on the first digit of a number or year and on the space before it, predicting more digits.
  - Fires on the first digit or leading space of a date or number (week ending [ ], 2[4], 1[5]), predicting digits.

## C008

PARENT: Fires on varied tokens in actor-career text ('a', 'the', 'and', 'previously', year digits); mixed.
CHILDREN:
  - Fires on 'the television', 'which', 'play', 'theatre' and the fragment ley in 'Ridley'; mixed.
  - Fires on assorted tokens: 'television', Rid-ley, 'the', 'UK', 'two', digits, adjectives like 'developed', 'cohesive', 'elaborate'; mixed.
  - Fires on modifiers in 'guest-starring role' and 'recurring role' phrases ('a [guest]', '[starring]', '[recurring]', 'had a'), and on 'to', 'who' in the Egyptian text; mostly an actor-credit unit.
  - Fires on 'The/a/the' after 'on' or 'appeared on', before the name of a TV show or programme; also 'come/came' in a few examples.

## C009

PARENT: Detects words inside song and album titles (Where We Are, Take Me Home, Kiss You, Mein Land, South of Heaven, Reign in Blood).
CHILDREN:
  - Detects words inside multi-word titles and names (Where We Are, Take Me Home, Kiss You, South of Heaven, Day-light, Don-key).
  - Detects interior subword pieces of titles and names (Merc-en-aries, Cas-ual-ty, Re-ign, W-aking, Unh-oly).

## C010

PARENT: Fires on the first digit inside a parenthetical inch or mm conversion after a metric measurement, and on "Yes" opening a quoted reply.
CHILDREN:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on discourse or function words in quoted speech or lyrics ("just", "if", "And", "my") plus stray spaces after "(".
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.
  - Fires on the first digit right after "(" in a mm or cm armour and gun size conversion.

## C011

PARENT: Fires on the last word that completes a quoted phrase or collocation ('Surfer [Girl]', 'done [before]', 'artistic [development]', 'personal [piety]', 'poet [historian]', 'called demons').
CHILDREN:
  - Fires on the last word that completes a quoted phrase or collocation ('Surfer [Girl]', 'poet [historian]', 'done [before]', 'Du [Fu]', 'personal [piety]', 'minor div-[inities]').
  - Fires on unit abbreviations in parentheses after a number (lb, km, mph, hp, yd, in, h, m).
  - Fires on the final noun that completes a phrase ('poetic utter-ance', 'thatched [hut]', 'love this [song]', 'artistic [development]', 'standby [alert]', 'Chinese [Virgil]').
  - Fires on the last word of a multi-word name or set phrase ('Kiss [You]', 'complete [symphony]', 'standby [alert]', 'Ferdinand [Max]'), where the phrase is about to end.

## C012

PARENT: Fires on a capital-letter first piece of a proper name or acronym (Wh-ishaw, B-oulter, Gid-o, I-mperial, I-HL) and predicts the next piece.
CHILDREN:
  - Fires on word-start fragments of long words (tur-ret, bec-ame, disl-ike, overh-ung, bel-ieved, conn-ected) and some name starts.
  - Fires on name fragments (K-ote-cha, Rif-enburg, Du, Le-onti, re @-@ entered); mostly the middle or first piece of a name.
  - Fires on name pieces (Wh-ishaw, Sh-iren, B-ai) and on the '@' marker in numbers and hyphens (4 @-@, 7 @,@), plus 'South'; mixed.

## C013

PARENT: Fires on the first digit of the converted value after a semicolon or opening bracket in speed and range specs (km / h ; 2-3 mph), and on 'Jan' or 'Christian' as the first name of a scholar.
CHILDREN:
  - Fires on the first digit of the converted figure in parentheses after 'long tons' or tonnes (17 @,@ 900 long tons ( 1-8 ...)). It marks a metric conversion for a weight.
  - Fires on the digit that starts or continues a converted number inside parentheses after metres, knots or feet and inches (29 metres ( 9-5 ft)).
  - Fires on the first digit or space inside the parenthesis after a metres, per-second or tons measurement, where the other unit is about to be written.
  - Fires on the first digit of the feet or inches value in parentheses after a metres figure, in ship and deck sizes. A weaker part fires on words after 'protect', 'freely' or 'growth' in aviation text.

## C014

PARENT: Mixed: double space after a sentence, opening quote after "said ,", "=" of headings, and "but" or "in" mid-clause.
CHILDREN:
  - Fires on closing quote marks, double spaces and "=" of headings at the end of a quoted or sentence unit; mixed.
  - Fires on sentence-closing marks: period, comma, question mark, closing quote, apostrophe and "=" of headings.
  - Mixed: "or", "only", "but", "not" qualifiers and "=" after a heading space.
  - Fires on closing and boundary marks: "=" heading, ")", dash, closing quote, double space and "mph".

## C015

PARENT: Fires on the space or first digit after "(" following a unit word (metres, tons, tonnes) in measurement conversions.
CHILDREN:
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on the opening parenthesis or first digit that begins a converted measurement after metres, tons or feet.
  - Fires on the first digit or space after "(" in conversions following tons, knots or km.
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.

## C016

PARENT: Fires on digits right after "(" in measurement conversions of armour, deck and ship sizes (mm, metres, inches).
CHILDREN:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.
  - Fires on the first digit right after "(" in long-ton and tonne conversions.

## C017

PARENT: Detects interior subword pieces of Japanese and other personal names (Fuk-ui, Ch-ū, Sh-ima, Tetsut-ar, Sh-ire, Esp-ito).
CHILDREN:
  - Mixed: relative or degree words ("who", "least", "to") and fragments of foreign names (shimа, Chū, Vevo).
  - Detects fragments of Japanese and foreign personal and place names (Satō Tetsutarō, Matsuo, Shizuo, Cattaro, Selivanov).
  - Detects the first token of a person's name, especially Chinese surnames and given names (Li, Wang, Wei, Zhang, Pe-i, Fang).
  - Detects interior fragments of personal names (Zhen, Zh, Ju, Ya, Sywell) and opening quote marks after "saying" or "noting".

## C018

PARENT: Fires on digits right after "(" in unit conversions after metres, kilograms or tonnes; also "CA" in ICAO.
CHILDREN:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on "(" and the first digits after it in conversions following kilograms, metres or tons.
  - Mixed: digits in ASIN codes and unit conversions, "sh" of romanized Chinese, and word endings (hangar, determinatives, Ogdoad).

## C019

PARENT: Fires on the first digit of a converted number after "(" following tons or knots.
CHILDREN:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on the first digit right after "(" in long-ton and tonne conversions.
  - Fires on digits within years and ranges (20[1]3, 8 to [4]) and on "@" in number separators.

## C020

PARENT: Mixed: double-space or sentence-end tokens and digits inside numbers; no single theme.
CHILDREN:
  - Mixed: sentence-start pronoun "It", sentence period or double space, and closing quote.
  - Fires on sentence-end period, double space and "=" of a heading boundary.
  - Mixed: double space after a heading, "of", "'s", "which" and name fragments (Kita, Jie).
  - Mixed: sentence-end period or closing quote with double space, and count words like "one", "three", "six".

## C021

PARENT: Fires on closing quote marks and apostrophes (a quotation mark or single quote right after a quoted word) and on the 's in possessives; also the second @ and 'ft'.
CHILDREN:
  - Fires on the apostrophe or curly quote right after a word (Fu ', Teddy ', don ’) and on 'War' in 'World War', plus broken CJK bytes.
  - Fires on a closing quotation mark or apostrophe after a quoted word, and on 's in possessives and plural -ars/-ima endings.
  - Fires on a quote mark or apostrophe that closes or follows a quoted word (Kiss You ", Prince ', don ’), and on word-middle pieces like -ast, -od.
  - Mixed: fires on the hyphen-@ in 'Ise @-@' headings, on the final piece of rare names (Z-aza, G-id, Nunn), and on shp.

## C022

PARENT: Fires on the last digits of numbers in ship specifications (28.65, 1,437, 42,67-1) and on closing parentheses after units.
CHILDREN:
  - Fires on a digit inside a figure with a thousands comma or decimal (7,500; 9,440; 1.737; 2 @,@ 200), predicting another digit.
  - Fires on a digit inside a measurement (95 ft, 104 ft, 9,440, 219 m, 414 ft, 71 ft), predicting another digit.
  - Fires on a digit inside a metric or imperial measurement in parentheses (39 ft, 3 ft 3 in), mostly the second digit, predicting a further digit.
  - Fires on a digit inside a number in a ship or athlete measurement (660 t, 18,289, 24,600 ft, 1.74 m), predicting more digits.

## C023

PARENT: Fires on the first or second digit of a parenthetical conversion after mm, kg or metres, plus "=" of headings.
CHILDREN:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on "(" and the first digits after it in conversions following kilograms, metres or tons.
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.

## C024

PARENT: Fires on the first digit (or space) right after an opening parenthesis that starts a converted unit value, mostly after a mm or cm measurement in ship armour and gun specs.
CHILDREN:
  - Fires on the first digit after '(' in a parenthesised conversion that follows an mm or centimetre armour or gun-calibre measurement.
  - Fires on the opening parenthesis or first digit of a converted value after knots, nautical miles, long tons or speed/range figures.
  - Fires on the '(' or the digits inside it after a weight or length figure (kilograms, metres, tons), where the converted unit is about to be given.
  - Fires on the first or second digit of the converted value in parentheses after a metres measurement (ship length, beam, deck sizes).

## C025

PARENT: Mixed formatting tokens: "(" after tons or kg, and "=" or spaces ending wiki section headings.
CHILDREN:
  - Fires on wikitext heading boundaries ("=" and the double space after a heading) and on possessive apostrophes.
  - Fires on sentence-end period, double space and "=" of a heading boundary.
  - Mixed punctuation: heading double space, "(" after units, commas in lists, closing quotes and sentence periods.
  - Fires on "(" after mm or tons, digits after it, and "=" ending wiki headings.

## C026

PARENT: Fires on the space, open parenthesis or first digit that starts a number in measurement and year contexts, predicting digits.
CHILDREN:
  - Fires on the first token of a number, often a digit or a space just after an opening parenthesis in a measurement or year, predicting a digit.
  - Fires on the start of a number inside a measurement or year, often the space, a parenthesis or the first digit, predicting digits.
  - Fires on the open parenthesis, space or first digit that begins a converted measurement ('metres ( [2]5 ft'), predicting digits.
  - Fires on the second digit of a year or number in a range (7-5-5, 7-4-4, 1934-3-7, 19-9), predicting another digit.

## C027

PARENT: Fires on a lone capital letter or short piece that starts a name (T-iffany, Zh-uang, J-ie, Z-ongwu, G-idō, K-ure, V-ena, O-la), predicting the next piece.
CHILDREN:
  - Fires on sh/ch/k consonant starts of romanized Chinese or Japanese words and names (Sh-iren, shèng, chéng, Ch-iyoda), predicting a vowel continuation.
  - Fires on the second or middle piece of a foreign name (Rami Y-ac, Kant-o, Zh-uang, Hy-ak-uman, Rub-in, Ge-b, H-ap-u), predicting the final piece.
  - Fires on a single capital letter or short fragment starting a proper name (T, W, Krist, K, B, AT, Port), predicting the rest of the name.
  - Fires on a single capital letter or short piece starting a Chinese or foreign name (B-aidicheng, Z-hen, Ku-izhou, Ch-ou, Ab-yss, S-layer), predicting the rest of the name.

## C028

PARENT: Fires on a capital letter or short piece that starts a name or title (St-ele, T-iffany, Ret-ired, Rem-inder, W-aking, H-athor, Sp-awn), predicting the rest.
CHILDREN:
  - Fires on a middle word of a song, album or tour title (Live While [We], Take [Me], Kiss [You], What Makes [You], Still [Reigning], [For] the Lions, Decade [of], [Come]back).
  - Fires on a capital-letter first piece of a name (R-ami, Sh-izuo, T-ak-, Ch-iyoda, Am-un, V-ader, t-inged) and unicode bytes, predicting the next piece.
  - Fires on word-start fragments (sh-rapnel, W-aking, Id-olator, scra-pped, rec-apture, exc-elling, dr-one, equ-ated) and apostrophes; broken-off word pieces.
  - Fires on the first fragment of an ordinary long English word (sl-ope, int-onational, spl-inters, inf-atuation, ret-arded, dis-liking) and predicts the rest of the word.

## C029

PARENT: Detects wikitext markup: "=" of headings, "@" of @-@ hyphens and @.@ decimals.
CHILDREN:
  - Fires on sentence-closing marks: period, comma, question mark, closing quote, apostrophe and "=" of headings.
  - Detects wikitext markup tokens such as "=" in section headings and "@" in @-@ hyphens, plus digits within large numbers.
  - Fires on closing and boundary marks: "=" heading, ")", dash, closing quote, double space and "mph".
  - Fires on markup pieces inside numbers and hyphens: "@" and "." in @.@, "-" in @-@, plus "of" after "velocity".

## C030

PARENT: Detects mid-word fragments of proper names and acronyms (IFP, RI, ICA, Alb-in, shi-se-i) and apostrophes after song titles.
CHILDREN:
  - Detects apostrophes, quote marks and dashes that close or mark a possessive or title (his father ', Kiss You ", don ’, – ranges).
  - Mixed punctuation inside tokens: possessive "'s", the thousands comma in "@,@", periods in abbreviations, and clause words like "who", "because it".
  - Detects an apostrophe or closing quote mark after a word (possessive, contraction, closing quote) and ordinal "th".
  - Mixed punctuation: apostrophes in possessives and contractions, "=" in headings, "@" in hyphens, and a space after "(".

## C031

PARENT: Fires on the unit after a number in a measurement (ft, mm, cm, km, metres), especially length or gun caliber.
CHILDREN:
  - Fires on the unit after a number in a measurement or speed (km, mph, ft, m, mi, metres); also on -ions in 'processions'.
  - Fires on the unit word after a number for distance, range or count (mi, miles, km, ft, nautical, views, hours).
  - Mixed: fires on commas in lists after a noun, on broken CJK bytes, and on mid-word pieces (Hot-ch, Tong-gu, Dao-ist).
  - Fires on the unit after a number in a measurement (feet, metres, ft), and on mid-word pieces (touch-down, D-ool, Z-ay).

## C032

PARENT: Fires on pieces of music-style phrases (uptem-po, up-beat, power pop, 'record's [second]', 'tribute', 'title', 'sounding') and on 'and' in lists; mixed.
CHILDREN:
  - Fires on pieces of music-style descriptions (uptem-po, up-beat, power pop, Motown-tinged, 'record's [second]', 'sounding', 'American') where the next word completes the genre or ordinal phrase.
  - Fires on varied words in review and chart text ('was', 'and', 'utterly', 'adorable', 'on', 'by', 'to', 'release', 'founded'); no shared idea.
  - Fires on nationality or group modifiers placed before a noun (English boy band, British cinema, UK and US versions, American thrash metal, second album), predicting the noun.
  - Fires on pieces of music-style descriptors (uptem-po, up-beat, power pop, Mot-own, surf-ing, speed metal, thrash metal, least favorite), where the next piece completes the genre.

## C033

PARENT: Fires on the head noun of a subject phrase just before its verb ('His [influence]', 'the [position]', 'high school [career]', 'religious [activity]', 'these [tasks]'), predicting a verb.
CHILDREN:
  - Fires on modifiers in 'guest-starring role' and 'recurring role' phrases ('a [guest]', '[starring]', '[recurring]', 'had a'), and on 'to', 'who' in the Egyptian text; mostly an actor-credit unit.
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on an adjective or participle that modifies a following noun, mostly 'role' ('a [starring] role', 'a [recurring] role', 'a [minor] role', '[diverse]', '[exalted]', '[important]').
  - Fires on prepositions and ordinals in sports record phrases ('17 [in] the [second] half', 'led the conference [in] [single] season', 'the [first]').

## C034

PARENT: Fires on the hyphen or @ symbol in the @-@ hyphen pattern (scholar @-@, little @-@) and on the second = of heading markers.
CHILDREN:
  - Fires on the @ symbol in @-@ or @.@ number/hyphen patterns, and on the first digit of a caliber number after 'calibre' (45-calibre [[3]]).
  - Fires on the second @ of the @-@ hyphen pattern and on heading = markers, plus on middle fragments of rare names (Indoch-ina, Mei Yao-c).
  - Mixed: fires on the second @ of @-@ after a word, and on mid-word pieces of names (Ky-ush-u, Saihok-ush) and on adverbs like 'temporary' or 'immediately'.
  - Fires on the @ in decimal and hyphen patterns (3 @.@ 429) and on the second = of heading markers such as '= = Charts ='.

## C035

PARENT: Fires on a subject pronoun, adverb or 'and' just before a verb in descriptive narrative ('It [was]', 'they', 'frequently', 'She [also]', 'he'), plus the first digit of a length.
CHILDREN:
  - Fires on passive or past-tense subjects and auxiliaries in military narrative ('the sisters [were]', 'Group [was]', 'launchers [were]', '[that]', 'ordered [them]', 'IJN [then]').
  - Fires on an adverb or auxiliary before a participle or verb ('collaboratively', 'was', 'finely', 'to be', 'previously', 'manually', 'tightly', 'being', 're-').
  - Fires on prepositions and determiners in sentences that place equipment on a ship ('mounted on [the]', 'three [on] each', 'forecastle [and]'), predicting a location or part noun.
  - Fires on position words and determiners in ship-layout descriptions ('fore-masts', 'mids-hips', 'Their', 'on [the]', 'aft', 'rebuilt', 'above [the]'); mixed.

## C036

PARENT: Fires on the first part of a hyphenated or compound technical term ('[high] @-@ tensile', '[surface] @-@ search', '[armor] @-@ piercing', '37 @-@ [millimeter]', 'maximum', 'lower', 'depth').
CHILDREN:
  - Fires on the first part of a compound technical term or its hyphen marker ('high @-@ tensile', 'muzzle velocity', 'surface @-@ search', 'torpedo tube', 'steam engines').
  - Fires on gun-caliber and unit tokens in armament lines ('12.7 [cm]', '25 [mm]', ')', 'cal-[iber]', '@', 'eight', 'three').
  - Fires on a modifier before a noun in technical ship and gun descriptions ('lacked [sufficient]', 'with a [higher]', '[excessive]', '[severe]', 'had [a]'), predicting a technical noun.
  - Fires on modifiers and quantities before equipment nouns in ship armament text ('a [maximum]', 'arm-[oured]', 'mounted in [three]', '[light] AA', '25 [mm]', '[secondary]', 'armored cruisers').

## C037

PARENT: Mixed: "(" after tons, double spaces, and past participles after "was" (born, died, captured, raised, intended).
CHILDREN:
  - Fires on the first digit or space after "(" in conversions following tons, knots or km.
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on the opening parenthesis or first digit that begins a converted measurement after metres, tons or feet.
  - Fires on "(" after tons or tonnes, the first digit after it, and "more" or "but" in comparisons.

## C038

PARENT: Fires on unit words and closing parentheses just after a measurement number ('20 [metres]', '[)]'), and on leading digits of figures.
CHILDREN:
  - Fires on a digit inside a metric or imperial measurement in parentheses (39 ft, 3 ft 3 in), mostly the second digit, predicting a further digit.
  - Fires on digits in speed figures (24 knots, 44 km/h, 20.5 knots, 28 mph), predicting another digit.
  - Fires on a digit inside a technical figure (muzzle velocity, height), on the space before a number, and on 'sh' for shaft horsepower.
  - Fires on a digit inside a measurement (95 ft, 104 ft, 9,440, 219 m, 414 ft, 71 ft), predicting another digit.

## C039

PARENT: Fires on the third digit of a four-digit year (20[1]x, 19[9]x), usually after a month or "in".
CHILDREN:
  - Detects chart-statement words: "Billboard", "week ending", "peaked at number ... on", "peaking at".
  - Fires on year digits in dates ("November 20[1]", "19[9]") and on "of" after "winter" or "summer".
  - Detects digits inside years and dates (2012, 2013, 25 April) and in ASIN codes.

## C040

PARENT: Fires on 'single', 'the', 'both', 'and', 'first', 'their' in sports sentences plus word pieces overh-, deb-; mixed.
CHILDREN:
  - Fires on prepositions and ordinals in sports record phrases ('17 [in] the [second] half', 'led the conference [in] [single] season', 'the [first]').
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on digits in sports scores and point totals (scoring 2-4, 1-7 points) and on 'single [season]' wording; mixed.

## C041

PARENT: Fires on lone capitals and name pieces (R-IAA, Mats-uda, I-se, MT-V, RH-9, Cape, I-CAO) and on commas and 'and' after names; mixed.
CHILDREN:
  - Fires on digits inside a catalog code, on unicode byte pieces, and on 'and', 'in', '(' and 'I'; mixed.
  - Fires on list separators after names ('Mark Strong [and]', 'Shane Zaza [,]', 'Kennedy [and]') and on role nouns after 'by' ('by [filmmaker]', 'theatre [critic]'); loosely a name-list unit.
  - Fires on commas, 'and' and 'by' that separate names in credit lists or lead into an agent ('written [by]', 'directed [by]', 'Falk [and] Yacoub'), plus sentence-final periods.

## C042

PARENT: Fires on the third digit of a 4-digit year after '19' (19[[1]], 19[[3]]) and on the unit 'ft' after a number.
CHILDREN:
  - Mixed: fires on the @ in numbers like 7 @.@ 62 (25 [[@]]), on 'more' or 'life' in loose phrases, and on word-end fragments.
  - Fires on the unit word after a number in a measurement (metres, ft, feet, inches, m), mostly length.
  - Fires on the @ in numbers like 53 @-@ and 7 @,@, and on the stem 'arm' (armour, armament) and the digit 9 in type numbers.
  - Fires on verbs and nouns that continue a phrase after a subject (focusing, intention, reassess, themes), and on the @ in big numbers.

## C043

PARENT: Fires on a subject or auxiliary before a past participle in narrative ('was', 'were', 'only', 'she', 'not', 'came', 'be'), predicting a participle or verb.
CHILDREN:
  - Fires on an auxiliary or adverb before a past participle in passive narrative ('was [not]', 'were', 'un-[successfully]', 'only', 'completely', 'ultimately').
  - Fires on 'was', 'were', 'to', 'be', 'again', 'would' that sit right before a verb in passive or infinitive constructions.
  - Fires on digits inside years and numbers (7-5-5, 16-2-4, 7-7-0) and on stray capitals such as I and Du.

## C044

PARENT: Fires on an adverb or pronoun just before the main verb ('they', 'then', 'finally', 'ever', 'frequently', 're-'), predicting a verb.
CHILDREN:
  - Fires on 'to', 'be', 'not', 'was', 'and' that come right before a verb in passive or infinitive constructions, predicting a verb.
  - Fires on an adverb or subject before the main verb ('briefly', 'frequently', 'then', 'next', 'they', 'finally', 'consciously', 'rarely'), predicting a verb.
  - Fires on a subject or adverb right before a verb ('Afterwards [they]', '[They]', 'track [also]', 'ultimately', 'ever', 'vessels', 'video', 'he'), predicting a verb.
  - Fires on the word before a past-tense verb or participle in narrative ('was [not]', 'Americans [finally]', 'was', 'unable [to]', 'briefly'), predicting a verb.

## C045

PARENT: Fires on 'were', 'be', 'then', 'could', 'would', 'to', 'never' that come right before a verb in passive or infinitive constructions.
CHILDREN:
  - Fires on 'was', 'were', 'to', 'be', 'again', 'would' that sit right before a verb in passive or infinitive constructions.
  - Fires on an auxiliary or adverb before a past participle in passive narrative ('was [not]', 'were', 'un-[successfully]', 'only', 'completely', 'ultimately').
  - Fires on 'to', 'be', 'not', 'was', 'and' that come right before a verb in passive or infinitive constructions, predicting a verb.
  - Fires on small function words in narrative ('Battle [of]', 'assigned [to]', 'at', 'to [be]', '[was]', '[were]', 'Kure [Naval]'); mixed.

## C046

PARENT: Fires on clause-closing punctuation: comma, period, semicolon, colon, closing parenthesis and double space.
CHILDREN:
  - Fires on commas, dashes and opening parentheses that begin an aside, including the thousands comma in "5 @,@ 000".
  - Fires on clause-closing punctuation: comma, period, closing parenthesis, semicolon, and double space after a heading.
  - Fires on closing punctuation after a clause or name: comma, period, closing parenthesis, closing quote.
  - Fires on sentence-boundary tokens (period, closing paren, double space) and on digits inside catalog codes like ASIN numbers.

## C047

PARENT: Detects date parts: day-of-month digits, "winter of", "summer of", month names, and the space before a day number.
CHILDREN:
  - Fires on month names and the day digit just before a month ("10 December", "25 April", "11 March").
  - Detects date words: month names, "of" after a season, "On", and the final digits of season years like 1998 – 9[9].
  - Fires on digits inside years and day numbers (1[9]..), and the space before a year.
  - Fires on year digits and the space before a year ("in 1926", "June 1[9]42"), plus "War" in "World War".

## C048

PARENT: Fires on sentence-ending punctuation, a closing quote or the double space after a full stop, mainly at the end of a sentence that quotes or cites a review or critic. It marks the end of a sentence.
CHILDREN:
  - Fires on the double space after a full stop, or on '=' signs and final punctuation that end a sentence or close a section heading, i.e. the point just before a new sentence or section.
  - Fires on the double space or full stop that ends a sentence, and on the word 'It' that starts the next sentence. Both mark a sentence boundary where a new subject is about to begin.
  - Fires on a boundary or continuation token: a double space after a colon or full stop, an apostrophe-s, 'of' after 'beam', or a mid-name piece (Admiral, Kita, Zhang J-ie). A mix of unrelated boundary and name pieces.
  - Fires on punctuation that introduces or continues a quotation or list: a colon before a quote or poem, a comma after 'commented that', a semicolon, a closing quote, and the double space after a heading.

## C049

PARENT: Fires on the first or second digit of a year (20[1]3, 1[9]23) and on apostrophes after nouns.
CHILDREN:
  - Mixed punctuation inside tokens: possessive "'s", the thousands comma in "@,@", periods in abbreviations, and clause words like "who", "because it".
  - Detects an apostrophe or closing quote mark after a word (possessive, contraction, closing quote) and ordinal "th".
  - Detects apostrophes, quote marks and dashes that close or mark a possessive or title (his father ', Kiss You ", don ’, – ranges).
  - Mixed punctuation: apostrophes in possessives and contractions, "=" in headings, "@" in hyphens, and a space after "(".

## C050

PARENT: Fires on the first piece of Japanese or Western personal names (Tak-ashi, Mats-, Fuk-ui, Yam-ashiro, Step-hens, Sav-an) and predicts the next piece; also 'had' in a few cases.
CHILDREN:
  - Fires on the surname or last part of a name after a first name (Liam [Payne], Burton [Watson], The [Guardian], Digital [Spy], Henry [Fischer], Horn-[ung]) and on 'he' after a named person.
  - Fires on the first piece of a place or personal name (Tak-ashi, Kal-amazoo, Ol-ly, Ok-u, Tur-ret, Am-ami, Sat-ō, Ot-ranto, Ul-f), predicting the next syllable.
  - Fires on 'to', 'be', 'not', 'was', 'and' that come right before a verb in passive or infinitive constructions, predicting a verb.

## C051

PARENT: Fires on 'was', 'were', 'also' and part nouns (gun, boiler) after a ship part in ship description; mixed.
CHILDREN:
  - Fires on 'It', 'was', 'and' that follow a bracketed measurement in ship dimension sentences ('...19 ft 8 in) [high]', 'It [was]').
  - Fires on the first part of a compound noun or the connector in equipment descriptions ('fire [fighting] foam', '[concrete] wall', '[twin]', '[storage] positions', 'a pair [of]').
  - Fires on words in sentences that place guns and parts on a ship ('forecastle [and]', 'protected [by]', 'super-structure', 'gun', 'forward', 'a', 'primary', 'two', 'eight'); loosely related.
  - Fires on modifiers and quantities before equipment nouns in ship armament text ('a [maximum]', 'arm-[oured]', 'mounted in [three]', '[light] AA', '25 [mm]', '[secondary]', 'armored cruisers').

## C052

PARENT: Fires on the second digit of a year (0 in 20xx, 9 in 19xx, 1 in 201x) after a month, date or season, where the digits of the year are being completed.
CHILDREN:
  - Fires on the month name (December, April, March, November, October) or a space or digit just after a day number in a date.

## C053

PARENT: Fires on a middle word of a title or name ('Live While [We]', 'What Makes [You]', 'Take [Me]', 'The [Selected] Poems', 'the [Second] Sino-Japanese', 'Second Intermediate').
CHILDREN:
  - Fires on a middle word of a multi-word title or name (Special [Naval], Live While [We], Take [Me], The [Selected] Poems, What Makes [You], Harry) and on blank space after a sentence.
  - Fires on structure tokens at line or section breaks: blank space after a sentence, '=' heading marks, closing quote or colon at the end of a line.
  - Fires on blank or spacing tokens after a sentence end or heading, and on stray pieces like 'sh' and unicode bytes; mixed.
  - Fires on structure tokens: blank space after a sentence or line, '=' heading marks, and the dot in decimals like 10 @.@ 4.

## C054

PARENT: Fires on the first piece of a multi-token foreign proper name (Japanese, Egyptian, Slavic) and predicts the next name fragment, as in Rami Y-ac or Ch-iy-oda.
CHILDREN:
  - Fires on the first piece of foreign or Western names (Mal-ik, Hor-an, Yam-ash-iro, Hō-sh-ō, Y-acoub, Bor-odino, B-ai) and on year digits.
  - Fires on incomplete word stems such as tur(ret), lic(ence), Wol(verines), Kal(amazoo), Et(ajima) and predicts the rest of the word.
  - Fires on name fragments (K-ote-cha, Rif-enburg, Du, Le-onti, re @-@ entered); mostly the middle or first piece of a name.
  - Fires on sh/ch/ts-type first pieces of romanized Japanese names and words (Amami Ō-sh-ima, Ts-ushima, Yokosuka, Yamashiro, Sh-iren) and predicts the next syllable.

## C055

PARENT: Detects sentence-initial or second-sentence tokens (He, It, She, also, In addition) and heading "=" markers.
CHILDREN:
  - Fires on wikitext heading boundaries ("=" and the double space after a heading) and on possessive apostrophes.
  - Fires on sentence-closing marks: period, comma, question mark, closing quote, apostrophe and "=" of headings.
  - Mixed: "or", "only", "but", "not" qualifiers and "=" after a heading space.
  - Detects sentence and section boundaries: double space, "=" heading marks, first word of a heading, and the dash in year headings.

## C056

PARENT: Detects interior subword pieces of foreign names and titles (Mei Yaochen, Shūreishū, Donkey Punch, Bunshū, Wadj-Wer).
CHILDREN:
  - Detects interior pieces of person and place names (Shū, Shi, Arn-ell, Chat-room, Yao-oc, Shiz-uo, Loft-us).
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.
  - Detects the first fragment of a personal name, especially Chinese names (Pei Di, Mei Ya, Su Shi, Du Z), and some Egyptian names.

## C057

PARENT: Mixed: "(" or first digit after metric units, plus fragments of names (Stroud, Tsukuba, Beach Boys, Tampa Bay) and the end of "Hot 100".
CHILDREN:
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.

## C058

PARENT: Fires on prepositions, 'both', ordinals and sport nouns in sports biography ('in [the] second half', 'in [both]', '[single] season', 'basketball'); loosely related.
CHILDREN:
  - Fires on prepositions and ordinals in sports record phrases ('17 [in] the [second] half', 'led the conference [in] [single] season', 'the [first]').
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on the second token of a multi-word proper name (Buffalo [Bull]s, Kalamazoo [Central], The [X] Factor, the Big [Ten], Boston [Bruins], Rockefeller), and on 'the' before such a name.

## C059

PARENT: Mostly fires on digits right after "(" in unit conversions, with a few name pieces (Ise) and a colon after a language label.
CHILDREN:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on digits in the inches part of a feet-inches conversion (e.g. "42 ft 8") and on colons after a language label.
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.

## C060

PARENT: Fires on contraction and possessive endings after an apostrophe (the s in 's, re in 're, t in n't) and on the second = of wiki heading markers.
CHILDREN:
  - Fires on the word part right after an apostrophe in contractions and possessives ('re, 've, 's, 't), plus the h in km / h.
  - Fires on the final = of a wiki section heading marker, and on mid-word fragments of rare names and long words (Rami, dega-ussing, Mus-well).
  - Fires on the 's or 're suffix after an apostrophe, on the second byte of broken CJK characters, and on the word 'room' after 'engine'.
  - Mixed: fires on the second = of section heading markers, on the word 'Factor' after 'The X', and on fragments of rare names (Petos-key, Cris-p).

## C061

PARENT: Fires on subject pronouns, 'also', 'the', 'single', 'season' in sports biography sentences; a loose mix with digits.
CHILDREN:
  - Fires on prepositions and ordinals in sports record phrases ('17 [in] the [second] half', 'led the conference [in] [single] season', 'the [first]').
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on digits in sports scores and point totals (scoring 2-4, 1-7 points) and on 'single [season]' wording; mixed.

## C062

PARENT: Mixed: "(" after units, commas after introductory phrases, and "and" joining clauses.
CHILDREN:
  - Fires on "(" after mm or tons, digits after it, and "=" ending wiki headings.
  - Fires on the first digit right after "(" in a mm or cm armour and gun size conversion.
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.
  - Fires on the opening parenthesis or first digit that begins a converted measurement after metres, tons or feet.

## C063

PARENT: Mixed: apostrophes after nouns or "don", "(" spaces, "=" heading marks and ";".
CHILDREN:
  - Mixed punctuation: apostrophes in possessives and contractions, "=" in headings, "@" in hyphens, and a space after "(".
  - Mixed punctuation inside tokens: possessive "'s", the thousands comma in "@,@", periods in abbreviations, and clause words like "who", "because it".
  - Detects apostrophes, quote marks and dashes that close or mark a possessive or title (his father ', Kiss You ", don ’, – ranges).
  - Detects an apostrophe or closing quote mark after a word (possessive, contraction, closing quote) and ordinal "th".

## C064

PARENT: Mixed: plural-possessive apostrophe after "album" or "band", sentence periods, and year digits.
CHILDREN:
  - Mixed punctuation inside tokens: possessive "'s", the thousands comma in "@,@", periods in abbreviations, and clause words like "who", "because it".
  - Mixed punctuation: apostrophes in possessives and contractions, "=" in headings, "@" in hyphens, and a space after "(".
  - Detects apostrophes, quote marks and dashes that close or mark a possessive or title (his father ', Kiss You ", don ’, – ranges).
  - Detects an apostrophe or closing quote mark after a word (possessive, contraction, closing quote) and ordinal "th".

## C065

PARENT: Fires on the hyphen after @ in @-@, on the second byte of broken CJK characters, on heading = markers, and on mid-word fragments (Pet-rol, gun-nery).
CHILDREN:
  - Mixed: fires on the hyphen of @-@ patterns, on name fragments (K-ier, Baid-), and on closing parentheses after a date or measurement.
  - Mixed: fires on the second = of heading markers, on suffix fragments (thatched, anthe-mic, a-irst), and on 'day' in 'late in the day'.
  - Fires on the second = of heading markers and the hyphen of @-@ patterns, on broken CJK bytes, and on the noun 'capacity' or 'hang(ar)'.
  - Fires on the hyphen in @-@ after a word and on middle or end fragments of rare names and words (Rid-ley, Yac-oub, solips-ism).

## C066

PARENT: Fires on the word before a verb or participle ('frequently', 'was', 'have', 'famously', 'been', 'be', 'who', 'later'), predicting a verb form.
CHILDREN:
  - Fires on subject words just before a verb ('he', 'who', 'it') and on nouns like Post, world, man, end; a mix.
  - Fires on passive or past-tense subjects and auxiliaries in military narrative ('the sisters [were]', 'Group [was]', 'launchers [were]', '[that]', 'ordered [them]', 'IJN [then]').
  - Fires on the surname or last part of a name after a first name (Liam [Payne], Burton [Watson], The [Guardian], Digital [Spy], Henry [Fischer], Horn-[ung]) and on 'he' after a named person.
  - Fires on an auxiliary or adverb before a participle in past-tense narrative ('was', 'were', 'not', 'then', 'largely', 'been', 'be', 'being').

## C067

PARENT: Detects wikitext formatting tokens: the second "=" of a heading marker and the "@" or "-" in @-@ hyphen sequences, plus "(".
CHILDREN:
  - Mixed: "(" after units and fragments of non-Latin or foreign name characters.
  - Fires on the space or first digit after "(" in unit conversions, and on dash ranges like "7 –".
  - Mixed punctuation: "(" after units, ":" after a language label, apostrophes, and "=" of headings.
  - Fires on the "(" or space that opens a conversion after a unit word (metres, tons, kilograms).

## C068

PARENT: Fires on mid-word subword pieces of proper names (Hy-ū-ga, Penguins, Whalers, Direction, Ise, Yay-oi).
CHILDREN:
  - Detects interior subword pieces of names and team nouns (Cam Ran, Pet-rid, Wolver-ines, Spir-its, Fly-ers, Hy-ū).
  - Detects interior subword pieces of proper names and titles (Yamash-iro, Hy-ū-ga, Tsuk-uba, Sur-fer, Re-ign, Kal-am-azoo).

## C069

PARENT: Fires on single letters, short fragments and final digits (d, v, ley, 1, 6, 7, 'a'); mixed.
CHILDREN:
  - Fires on assorted tokens: 'television', Rid-ley, 'the', 'UK', 'two', digits, adjectives like 'developed', 'cohesive', 'elaborate'; mixed.
  - Fires on the word after 'role in/on' or 'appeared in' (the, television, two, ten), on year digits like 200-7, and on a capital V; a mix of unrelated parts.
  - Fires on 'the television', 'which', 'play', 'theatre' and the fragment ley in 'Ridley'; mixed.

## C070

PARENT: Fires on the second or later digit of a number in speed or time measurements (16 knots, 28 mph, 16:40), predicting another digit.
CHILDREN:
  - Fires on digits in speed figures (24 knots, 44 km/h, 20.5 knots, 28 mph), predicting another digit.
  - Fires on digits in times, durations and scores (3:04, 18:xx, 0-7, 7-3 victory).
  - Fires on the first or second digit of a number in a speed, range or percentage figure (16 knots, 320 km, 770 m/s, 60 per cent).
  - Fires on the first digit of a large count in sales, census or shipment facts ('sales of [7]', 'shipments of [5]0', 'recorded [5]2', 'more than [1]'), predicting more digits.

## C071

PARENT: Fires on words in lists of media or sports nouns ('sports', 'radio', 'television', 'one season', 'X Factor UK'); a loose mix.
CHILDREN:
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on list separators and verbs in sports biography ('basketball [,] [and]', 'in', 'He [played]', '[various]', 'Arthur [Hill]', 'National'); mixed.
  - Fires on sport or league name words ('Big Ten [football]', 'high school [basketball]', 'College [Football]', 'Wolver-ines', 'Ath-letic', '[season]') where the sport noun is expected or just given.

## C072

PARENT: Fires on "@" or "-" in @-@ hyphen markup and on closing ")" or quote marks.
CHILDREN:
  - Fires on markup pieces inside numbers and hyphens: "@" and "." in @.@, "-" in @-@, plus "of" after "velocity".
  - Mixed: "@" or "-" in hyphen markup, second words of names (Du Fu, South of Heaven, New Zealand) and possessive apostrophes.
  - Mixed formatting tokens: "@" in "@,@" number separators, "=" and spaces of headings, and the unit "mph".

## C073

PARENT: Fires on pre-noun modifiers and connectors ('two', 'Naval', 'British', 'maximum', 'other', 'and', 'of', 'largest'); mixed.
CHILDREN:
  - Fires on ordinals and nationality or prefix words before a unit or class noun ('2-[nd]', '1-[st]', '[Third]', '[Second]', '[Fourth]', '[American]', '[Italian]', '[Anti]', '[pre]', 'Novara @-@ [class]').
  - Fires on past-tense verbs or modifiers in military narrative ('received', 'was', 'severe', 'trained', 'qualified', 'of'); mixed.
  - Fires on 'During' and the 'the' that follows, or on 'until' and 'of', words that open a time phrase, and on closing parentheses; loosely a time-phrase opener.
  - Fires on ordinal words and suffixes in unit names (3-rd Fleet, 1-st Fleet, 2-nd Division, 5-th Fleet, Third, Fourth), predicting 'Fleet' or 'Division'.

## C074

PARENT: Fires on digits in sports scores and on words in 'single season', 'including', 'the second half', 'and'; same mix as other sports-record units.
CHILDREN:
  - Fires on digits in sports scores and point totals (scoring 2-4, 1-7 points) and on 'single [season]' wording; mixed.
  - Fires on digits inside numbers and on 'million' after a figure, predicting further digits or a unit.
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.

## C075

PARENT: Fires on the first word or piece of a multi-word title, band, song or album name after a quote or comma (Take Me Home, Just Dance, South of Heaven, Day-light Robbery, Don-key Punch). It predicts the rest of the title.
CHILDREN:
  - Fires on the middle word or piece of a multi-word title or name (Take Me Home, Kiss You, Daylight Robbery, South of Heaven, Where We Are). It marks that a title is in progress.
  - Fires inside a person's name or in a review sentence about a named actor or athlete (Ben Whishaw, Zalapski, Esposito, Shawn, 'Boulter starred as'). It marks a name that is being spelled out or a person-focused verb.

## C076

PARENT: Fires on an auxiliary or adverb before a past participle in passive or narrative text ('was', 'were', 'be', 'never', 'previously', 'then', 'ultimately').
CHILDREN:
  - Fires on an adverb or auxiliary before a participle or verb ('collaboratively', 'was', 'finely', 'to be', 'previously', 'manually', 'tightly', 'being', 're-').
  - Fires on passive or past-tense subjects and auxiliaries in military narrative ('the sisters [were]', 'Group [was]', 'launchers [were]', '[that]', 'ordered [them]', 'IJN [then]').
  - Fires on an auxiliary or adverb before a past participle in passive narrative ('was [not]', 'were', 'un-[successfully]', 'only', 'completely', 'ultimately').
  - Fires on the word before a past-tense verb or participle in narrative ('was [not]', 'Americans [finally]', 'was', 'unable [to]', 'briefly'), predicting a verb.

## C077

PARENT: Fires on the first digit of a year (1 or 2) or the space before it, mostly after a month and day in a date (on 27 January 2, October 1).
CHILDREN:
  - Fires on the digit that starts or continues a year (1, 9, 0), and on 'of' after a season or 'morning' (summer of, winter of, morning of), where a date or year follows.
  - Fires on the digit '1', '0', '9' or '2' of a year after a month name (November 20-, December 8, 7-), and on 'of' after a season (winter of, June of).
  - Fires on the first or second digit of a four-digit year, or on the space right before it, after 'in', a month or a day (in 1926, June 1-9, 17 January 2-0).
  - Fires on the beginning digit of a year (1, 2, 7) or the space before it, mostly after a month or a day in a historical date.

## C078

PARENT: Fires on the first digit or the "(" that opens a unit conversion after mm, metres, tons or centimetres.
CHILDREN:
  - Fires on "(" and the first digits after it in conversions following kilograms, metres or tons.
  - Fires on the first digit right after "(" in a mm or cm armour and gun size conversion.
  - Fires on the first digit or space after "(" in conversions following tons, knots or km.
  - Fires on the opening parenthesis or first digit that begins a converted measurement after metres, tons or feet.
