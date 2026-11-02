# Family judging (L15g_input_K48)

Each case has one PARENT unit and two sets of other units (X and Y). Exactly one set holds units that the
parent was found to cover. Pick the set for which the parent is best described as a broader, coarser
category that includes the units in the set. Give a confidence 1 (guess) to 5 (sure).


## F1

PARENT: Fires on the third digit of a four-digit year (20[1]x, 19[9]x), usually after a month or "in".
Set X:
  - Detects interior and final fragments of Japanese and Egyptian personal names (Chiaki, Tetsutarō, Shirō, Shizuo, Amenhotep), plus a few small function words.
  - Detects prepositions and articles after verbs of participation or role ("participated in", "served as the", "one of the", "part of").
  - Detects interior pieces of foreign proper names (Sai-h, Gah-ō, Chang, Bai Ju, Hy-ū, Szent Ist, Du-at).
Set Y:
  - Detects chart-statement words: "Billboard", "week ending", "peaked at number ... on", "peaking at".
  - Fires on year digits in dates ("November 20[1]", "19[9]") and on "of" after "winter" or "summer".
  - Detects digits inside years and dates (2012, 2013, 25 April) and in ASIN codes.

## F2

PARENT: Fires on the space or first digit after "(" following a unit word (metres, tons, tonnes) in measurement conversions.
Set X:
  - Detects exception or limiting qualifiers such as "except (for)", "particularly", "especially", "only", "almost", "some" that narrow a statement.
  - Mixed function words after passive or modal verbs ("held up by", "coupled with", "can be", "contend with", "complicated by").
  - Detects passive reporting frames: "is thought to", "is noted for", "was first mentioned in".
  - Fires on digit pieces inside multi-digit numbers and page numbers, plus a few prepositions.
Set Y:
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on the opening parenthesis or first digit that begins a converted measurement after metres, tons or feet.
  - Fires on the first digit or space after "(" in conversions following tons, knots or km.
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.

## F3

PARENT: Fires on the first digit or the "(" that opens a unit conversion after mm, metres, tons or centimetres.
Set X:
  - Detects interior pieces of proper names and titles (I-se, Sats-uma, Ts-ushima, Tu Fu, Kiss You, International Civil Aviation).
  - Mixed: spaces after numbers in specs, noun heads like "guns", "cruisers", and words like "every", "except", "allowed".
  - Mixed: ordinal suffixes (1st, 2nd, 3rd), "so", "and" and digits in times.
  - Mixed: opening quote after a colon, prepositions "from/through/of", "just", "Yeah", "often".
Set Y:
  - Fires on "(" and the first digits after it in conversions following kilograms, metres or tons.
  - Fires on the first digit right after "(" in a mm or cm armour and gun size conversion.
  - Fires on the first digit or space after "(" in conversions following tons, knots or km.
  - Fires on the opening parenthesis or first digit that begins a converted measurement after metres, tons or feet.

## F4

PARENT: Fires on clause-closing punctuation: comma, period, semicolon, colon, closing parenthesis and double space.
Set X:
  - Fires on prepositions and linking words that introduce a complement ("with", "by", "is", "was", "a", "of", colon).
  - Mixed: nouns or pronouns after a setup phrase ("in normal times", "raised the stakes", "this time it", "hits it").
  - Mixed nouns and prepositions that finish a place or route phrase (along, entrance, route, room, coast, home, installations, private).
  - Detects prepositions that complete a verb or adjective ("known for", "differed from", "range from", "on"), and "same" in "the same".
Set Y:
  - Fires on commas, dashes and opening parentheses that begin an aside, including the thousands comma in "5 @,@ 000".
  - Fires on clause-closing punctuation: comma, period, closing parenthesis, semicolon, and double space after a heading.
  - Fires on closing punctuation after a clause or name: comma, period, closing parenthesis, closing quote.
  - Fires on sentence-boundary tokens (period, closing paren, double space) and on digits inside catalog codes like ASIN numbers.

## F5

PARENT: Fires on the "(" or first digit that opens a unit conversion after miles, tons, inches or mm.
Set X:
  - Fires on a possessive apostrophe or "'s" after a noun (One Direction 's, Beach Boys ', gods '), and on "his" or the noun after it.
  - Mixed: apostrophes in contractions and possessives, and prepositions or conjunctions like "by", "until", "unless", "under".
  - Mixed: dash in year ranges, final year digits, possessive "his" and "continue to"; no single theme.
  - Mixed: verbs and prepositions of seeking, request or purpose ("begging help", "Should I", "funding to proceed with", "hoping for a").
Set Y:
  - Fires on the first digit or space after "(" in conversions following tons, knots or km.
  - Fires on the first digit right after "(" in a mm or cm armour and gun size conversion.
  - Fires on "(" after mm or tons, digits after it, and "=" ending wiki headings.
  - Fires on "(" after tons or tonnes, the first digit after it, and "more" or "but" in comparisons.

## F6

PARENT: Fires on mid-word subword pieces of proper names (Hy-ū-ga, Penguins, Whalers, Direction, Ise, Yay-oi).
Set X:
  - Detects quantity words and numbers ("more", "own", "three", "one", "few", "all three", "only 6") that count or limit things.
  - Mixed: list commas and "and", modal words ("might", "enough"), "know", and fragments after "However much".
Set Y:
  - Detects interior subword pieces of names and team nouns (Cam Ran, Pet-rid, Wolver-ines, Spir-its, Fly-ers, Hy-ū).
  - Detects interior subword pieces of proper names and titles (Yamash-iro, Hy-ū-ga, Tsuk-uba, Sur-fer, Re-ign, Kal-am-azoo).

## F7

PARENT: Fires on "@" or "-" in @-@ hyphen markup and on closing ")" or quote marks.
Set X:
  - Detects words introducing a reason, purpose or contrast ("financial", "economic", "due to", "increase", "but", "and") in explanatory narrative.
  - Detects attribution and estimate phrases: "according to", "estimates", "it has been estimated", and the comma after "not available".
  - Mostly detects the @ or - in "@-@" hyphen markup (re @-@ entered, sold @-@ out), plus verb stems like "star".
Set Y:
  - Fires on markup pieces inside numbers and hyphens: "@" and "." in @.@, "-" in @-@, plus "of" after "velocity".
  - Mixed: "@" or "-" in hyphen markup, second words of names (Du Fu, South of Heaven, New Zealand) and possessive apostrophes.
  - Mixed formatting tokens: "@" in "@,@" number separators, "=" and spaces of headings, and the unit "mph".

## F8

PARENT: Mixed: double space after a sentence, opening quote after "said ,", "=" of headings, and "but" or "in" mid-clause.
Set X:
  - Fires on closing quote marks, double spaces and "=" of headings at the end of a quoted or sentence unit; mixed.
  - Fires on sentence-closing marks: period, comma, question mark, closing quote, apostrophe and "=" of headings.
  - Mixed: "or", "only", "but", "not" qualifiers and "=" after a heading space.
  - Fires on closing and boundary marks: "=" heading, ")", dash, closing quote, double space and "mph".
Set Y:
  - Detects "even" and the "such an extent that" intensifier construction, plus "to be/that" after result phrases.
  - Mixed: section-title and noun-phrase starts ("A second favourite", "Moral engagement", "The design", "In the fall", "accompanying").
  - Fires on a number that gives a count of items before a noun, such as "eight [4] cm guns" or "four [4]" gun mounts.
  - Mixed: "of", "for", "the", "known", "exact", "which" after nouns; also lyric words "did", "ever", "know".

## F9

PARENT: Fires on the first or second digit of a year (20[1]3, 1[9]23) and on apostrophes after nouns.
Set X:
  - Mixed punctuation inside tokens: possessive "'s", the thousands comma in "@,@", periods in abbreviations, and clause words like "who", "because it".
  - Detects an apostrophe or closing quote mark after a word (possessive, contraction, closing quote) and ordinal "th".
  - Detects apostrophes, quote marks and dashes that close or mark a possessive or title (his father ', Kiss You ", don ’, – ranges).
  - Mixed punctuation: apostrophes in possessives and contractions, "=" in headings, "@" in hyphens, and a space after "(".
Set Y:
  - Fires on number-formatting pieces: "@" in "@,@" separators, en dash in ranges, digits and "y" in yd.
  - Mixed causal and purpose words ("because", "due", "grounds", "in order to", "attributed", "for", "leading up to") with some adjectives like "mere" and "pure".
  - Fires on the first or second digit of a day-of-month number in dates (2[7], 3[0], 1[8]).
  - Mixed: unit-conversion digits, fragments of names (Mal-ik, Pet-), "Home" in Take Me Home, and "for" in quotes.

## F10

PARENT: Detects date parts: day-of-month digits, "winter of", "summer of", month names, and the space before a day number.
Set X:
  - Fires on month names and the day digit just before a month ("10 December", "25 April", "11 March").
  - Detects date words: month names, "of" after a season, "On", and the final digits of season years like 1998 – 9[9].
  - Fires on digits inside years and day numbers (1[9]..), and the space before a year.
  - Fires on year digits and the space before a year ("in 1926", "June 1[9]42"), plus "War" in "World War".
Set Y:
  - Mixed: "to", "had", "not", "@" and name fragments (Leont-i); no single theme.
  - Detects transitive verb stems awaiting an object (wounded, hoisted, counter, pepper, put forward, understand, release, top).
  - Detects contrast and negation words that turn a clause: "but", "however", "no", "not", "just", "could", and the comma before them.
  - Fires on abstract head nouns after a determiner or possessive (reputation, practice, translations, period, power, actions).

## F11

PARENT: Fires on the first digit of a converted number after "(" following tons or knots.
Set X:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on the first digit right after "(" in long-ton and tonne conversions.
  - Fires on digits within years and ranges (20[1]3, 8 to [4]) and on "@" in number separators.
Set Y:
  - Detects adjectives or participles that modify a following noun (favorable, creditable, modern, trained, qualified, expensive, better, unjust).
  - Detects nouns and verbs of attitude that expect a complement ("concern for", "loyalty to", "desire to", "emphasized the", "simplicity").
  - Fires on mid-phrase function words (his, of, the, as, a, for) in praise or purpose phrases like "to serve his country as"; mixed with "raise money for".
  - Detects the second word of fixed proper phrases such as "Imperial Japanese", "Second Sino-Japanese War", "World War", "2nd".

## F12

PARENT: Mixed: plural-possessive apostrophe after "album" or "band", sentence periods, and year digits.
Set X:
  - Mixed: "Selected", "accidentally", "that", "addition", "death", "present", "but", "used", "was"; no single theme.
  - Detects prepositions and articles in chart-position statements ("debut at number", "peaked on the", "entered at number 152 in the") and "has".
  - Mixed: ASIN code characters and fragments of names (Fuk-ui, Sh-iz, Dim, Mon, Bill-board).
  - Detects intensifier or scope words like "even", "all", "any", "anything" in superlative or comparison phrases ("bigger than anything", "highest of all").
Set Y:
  - Mixed punctuation inside tokens: possessive "'s", the thousands comma in "@,@", periods in abbreviations, and clause words like "who", "because it".
  - Mixed punctuation: apostrophes in possessives and contractions, "=" in headings, "@" in hyphens, and a space after "(".
  - Detects apostrophes, quote marks and dashes that close or mark a possessive or title (his father ', Kiss You ", don ’, – ranges).
  - Detects an apostrophe or closing quote mark after a word (possessive, contraction, closing quote) and ordinal "th".

## F13

PARENT: Fires on digits right after "(" in measurement conversions of armour, deck and ship sizes (mm, metres, inches).
Set X:
  - Detects time-period words and the preposition before them: season and month names, "during", "normal times", "early November", "festival", "annual".
  - Mixed: "present", ordinals, and hedging adverbs ("especially", "particularly", "considered", "about", "called").
  - Fires on nouns or adjectives after "was/be/the" (born, due, natural, life, range, center); no single theme.
  - Mixed: the second part of a name or noun phrase (Chinese, Du, Presley, Mouth, Blood, water); no single theme.
Set Y:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.
  - Fires on the first digit right after "(" in long-ton and tonne conversions.

## F14

PARENT: Detects sentence-initial or second-sentence tokens (He, It, She, also, In addition) and heading "=" markers.
Set X:
  - Fires on wikitext heading boundaries ("=" and the double space after a heading) and on possessive apostrophes.
  - Fires on sentence-closing marks: period, comma, question mark, closing quote, apostrophe and "=" of headings.
  - Mixed: "or", "only", "but", "not" qualifiers and "=" after a heading space.
  - Detects sentence and section boundaries: double space, "=" heading marks, first word of a heading, and the dash in year headings.
Set Y:
  - Fires on auxiliary or linking function words ("be", "was", "had", "as", "more") in mid-clause where a verb or participle is expected next; mixed with stray digits.
  - Mixed: word endings (watertight, hangar), "=" and open quote after a heading or "stated", and "deck"; no single theme.
  - Mixed: roles and verbs for critics and reporters (contributor, editor, commentary, favoured) and aviation training nouns.
  - Mixed: ASIN code digits, Chinese pinyin and name pieces (Li, rm), "'s" and "=" heading marks.

## F15

PARENT: Detects mid-word fragments of proper names and acronyms (IFP, RI, ICA, Alb-in, shi-se-i) and apostrophes after song titles.
Set X:
  - Detects apostrophes, quote marks and dashes that close or mark a possessive or title (his father ', Kiss You ", don ’, – ranges).
  - Mixed punctuation inside tokens: possessive "'s", the thousands comma in "@,@", periods in abbreviations, and clause words like "who", "because it".
  - Detects an apostrophe or closing quote mark after a word (possessive, contraction, closing quote) and ordinal "th".
  - Mixed punctuation: apostrophes in possessives and contractions, "=" in headings, "@" in hyphens, and a space after "(".
Set Y:
  - Mixed: unit words after numbers ("second", "kW", "metres") and abstract nouns (expense, time, desire); no single theme.
  - Fires on the last digit of three-digit counts (of men, guns, points) such as 11[1], 63[4], 10[4], plus "to" and "the".
  - Mixed: later digits inside large numbers (36 @,@ 500; 7 @,@ 500) and reporting verbs like "appreciated", "noted", "dismissed".
  - Detects ranking and quantity words in statistics ("number two", "top ten", "per", "third", "set of").

## F16

PARENT: Fires on digits right after "(" in unit conversions after metres, kilograms or tonnes; also "CA" in ICAO.
Set X:
  - Detects negation and denial words: "not", "never", "none", "no", "unable to", "than", "lacked".
  - Mixed: modifiers like "new", "newly", "formed", digits in years and spaces after "to"; no single theme.
  - Mixed: verbs and prepositions after a subject ("is noted for", "led", "includes", "would", "who") and colons.
  - Mixed: year digits, "issue", "important", "few", sentence periods; no single theme.
Set Y:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on "(" and the first digits after it in conversions following kilograms, metres or tons.
  - Mixed: digits in ASIN codes and unit conversions, "sh" of romanized Chinese, and word endings (hangar, determinatives, Ogdoad).

## F17

PARENT: Mixed: apostrophes after nouns or "don", "(" spaces, "=" heading marks and ";".
Set X:
  - Mixed punctuation: apostrophes in possessives and contractions, "=" in headings, "@" in hyphens, and a space after "(".
  - Mixed punctuation inside tokens: possessive "'s", the thousands comma in "@,@", periods in abbreviations, and clause words like "who", "because it".
  - Detects apostrophes, quote marks and dashes that close or mark a possessive or title (his father ', Kiss You ", don ’, – ranges).
  - Detects an apostrophe or closing quote mark after a word (possessive, contraction, closing quote) and ordinal "th".
Set Y:
  - Mixed: ordinals ("sixth"), duration nouns ("years"), participles and "was"; no single theme.
  - Fires on the last digits of years inside parenthetical date ranges (1926 – 1994, c. 2055 – 1650) and on "in" after feet.
  - Fires on mid-sentence function words (were, had, maintain, at, of, up) in narrative prose and on the year digit "1"; no single theme.
  - Mixed: "(" after tons, spatial prepositions ("at", "on"), and nouns like "course", "total", "magazines".

## F18

PARENT: Mostly fires on digits right after "(" in unit conversions, with a few name pieces (Ise) and a colon after a language label.
Set X:
  - Mixed: ASIN code letters, double space after a colon, "or" between numbers, and "even" or "such" in asides.
  - Fires on the last digit of a year (201[0], 200[4], 75[9]) and on "always", "but", "has" in other contexts.
  - Mixed: phrase-final nouns (crowd, readers, sailors, Wales, family, amateur) and "a", "of", "as" before them.
  - Detects modal and necessity words: "should", "need(ed)", "necessary", "could", "will", "had to", "inevitably".
Set Y:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on digits in the inches part of a feet-inches conversion (e.g. "42 ft 8") and on colons after a language label.
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.

## F19

PARENT: Mixed: double-space or sentence-end tokens and digits inside numbers; no single theme.
Set X:
  - Mixed: sentence-start pronoun "It", sentence period or double space, and closing quote.
  - Fires on sentence-end period, double space and "=" of a heading boundary.
  - Mixed: double space after a heading, "of", "'s", "which" and name fragments (Kita, Jie).
  - Mixed: sentence-end period or closing quote with double space, and count words like "one", "three", "six".
Set Y:
  - Mixed scope and limiting words ("any", "enough", "never", "without", "little", "not", "smaller") plus a few nouns.
  - Fires on prepositions and relatives after verbs of motion, place or time ("stretched forward to", "died in", "myth in which", "before").
  - Mixed function words in a translation passage ("need", "we", "why", "which", "complex", "with"); no single theme.
  - Mixed: main verbs and verb endings after a subject or "to" ("failed to", "able to", "represented", "depicted", "saw").

## F20

PARENT: Fires on the first digit inside a parenthetical inch or mm conversion after a metric measurement, and on "Yes" opening a quoted reply.
Set X:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on discourse or function words in quoted speech or lyrics ("just", "if", "And", "my") plus stray spaces after "(".
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.
  - Fires on the first digit right after "(" in a mm or cm armour and gun size conversion.
Set Y:
  - Mixed noun pieces and word endings (Jail, skiers, Turret, track, office, magazine, Donkey) inside lists or titles; no single theme.
  - Detects date-setting words: month names, "winter of", "in October", plus "for" after a time word and some name pieces.
  - Detects words inside titles and names (Cam Ran, What Makes You, Where We Are, Now That, Social 50, The Song of the).
  - Mixed: infinitive "to", "for" and ":" or an opening quote after verbs of request, purpose or reporting ("request to", "called for", "opened as follows :").

## F21

PARENT: Mixed: "(" after units, commas after introductory phrases, and "and" joining clauses.
Set X:
  - Detects adjectives and participles that modify a noun ("qualified", "trained", "informed", "loose", "indirect", "smaller", "firm").
  - Mixed: "known", "experience", feet digits, Hy-ū fragments and adjectives like "scholarly", "latest", "current".
  - Mixed: "or", "exact", "enough", "any", "likely", "serve" and a digit; limiting or alternative words.
  - Detects subordinating conjunctions that start a condition or time clause: "until", "if", "when", "unless", "for".
Set Y:
  - Fires on "(" after mm or tons, digits after it, and "=" ending wiki headings.
  - Fires on the first digit right after "(" in a mm or cm armour and gun size conversion.
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.
  - Fires on the opening parenthesis or first digit that begins a converted measurement after metres, tons or feet.

## F22

PARENT: Detects words inside song and album titles (Where We Are, Take Me Home, Kiss You, Mein Land, South of Heaven, Reign in Blood).
Set X:
  - Detects words inside multi-word titles and names (Where We Are, Take Me Home, Kiss You, South of Heaven, Day-light, Don-key).
  - Detects interior subword pieces of titles and names (Merc-en-aries, Cas-ual-ty, Re-ign, W-aking, Unh-oly).
Set Y:
  - Mixed: name endings (forerunner, Wolver-ines, Sab-res, IHL), possessive 's, and "to" or "of" in phrases.
  - Mixed: Chinese character byte pieces, name endings, and "of the" or "that" after a title.

## F23

PARENT: Detects wikitext markup: "=" of headings, "@" of @-@ hyphens and @.@ decimals.
Set X:
  - Fires on sentence-closing marks: period, comma, question mark, closing quote, apostrophe and "=" of headings.
  - Detects wikitext markup tokens such as "=" in section headings and "@" in @-@ hyphens, plus digits within large numbers.
  - Fires on closing and boundary marks: "=" heading, ")", dash, closing quote, double space and "mph".
  - Fires on markup pieces inside numbers and hyphens: "@" and "." in @.@, "-" in @-@, plus "of" after "velocity".
Set Y:
  - Mixed nouns and adverbs after negation or determiners (translation, computer, exact, translators, fully, perfectly).
  - Detects coordinating words that offer alternatives or contrast ("or", "but", "one", "and") as in "one or more", "6 or 8", "either ... or".
  - Fires on a preposition or opening parenthesis ("in", "to", "for", "on", "from", "(") that follows a verb or noun and expects a place, date or name next.
  - Mixed function words continuing a noun phrase ("series of", "first of a", "whose", "as ... they", "its", "with").

## F24

PARENT: Fires on the first or second digit of a parenthetical conversion after mm, kg or metres, plus "=" of headings.
Set X:
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on "(" and the first digits after it in conversions following kilograms, metres or tons.
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.
Set Y:
  - Fires on "and" joining verb clauses ("failed ... and", "collided with and sank"), plus some digits; mixed.
  - Detects the final noun or pronoun that ends a prepositional phrase (ships, carriers, deck, desk, bulkheads, him, her, his).
  - Mixed: "You" in "Kiss You", hyphen markup "@-@", "all", "be", and name fragments; no single theme.
  - Mixed: month names, ordinals, passive past verbs (rejected, removed, failed) and "number [1]"; no single theme.

## F25

PARENT: Mixed formatting tokens: "(" after tons or kg, and "=" or spaces ending wiki section headings.
Set X:
  - Fires on the final digit of a multi-digit number such as a chart position, year or measurement (46, 83, 1994).
  - Mixed: articles, commas, "and", "during", and fragments of ship and place names.
  - Detects subword pieces of foreign or technical terms (lǜshi, Sansh-ik, Stroud, Vickers, kans, Ba-rr).
  - Detects prepositions that complete a noun or verb of origin or influence ("from", "on", "in", "of" after influential, traditions, copy).
Set Y:
  - Fires on wikitext heading boundaries ("=" and the double space after a heading) and on possessive apostrophes.
  - Fires on sentence-end period, double space and "=" of a heading boundary.
  - Mixed punctuation: heading double space, "(" after units, commas in lists, closing quotes and sentence periods.
  - Fires on "(" after mm or tons, digits after it, and "=" ending wiki headings.

## F26

PARENT: Detects interior subword pieces of foreign names and titles (Mei Yaochen, Shūreishū, Donkey Punch, Bunshū, Wadj-Wer).
Set X:
  - Detects interior pieces of person and place names (Shū, Shi, Arn-ell, Chat-room, Yao-oc, Shiz-uo, Loft-us).
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.
  - Detects the first fragment of a personal name, especially Chinese names (Pei Di, Mei Ya, Su Shi, Du Z), and some Egyptian names.
Set Y:
  - Mixed: ASIN code characters, a space after "km ;", ":" and "=" heading marks, and stray function words.
  - Mixed: opening quotes after colons, "his", "and", "one", "its", and verbs like "pass into"; no clear theme.
  - Mixed: "of", "to", "than", "only" in comparison and addition frames ("Most of", "In addition to", "other than", "not only").
  - Fires on auxiliary verbs "has been", "is" and the word after them in perfect or passive statements, plus "present" and some year digits.

## F27

PARENT: Detects wikitext formatting tokens: the second "=" of a heading marker and the "@" or "-" in @-@ hyphen sequences, plus "(".
Set X:
  - Detects dimension terms after a measurement: "wide", "high", "beam", "elevation", "feet", "metres".
  - Fires on digits and the decimal "." inside numbers written with @.@ formatting, plus ASIN code digits and "t" in a contraction.
  - Fires on the third digit of a year (20[1]3, 19[9]4, 19[8]8) and on Chinese character byte fragments.
  - Mixed prepositions and career nouns (in, of, season, operations, career, sports).
Set Y:
  - Mixed: "(" after units and fragments of non-Latin or foreign name characters.
  - Fires on the space or first digit after "(" in unit conversions, and on dash ranges like "7 –".
  - Mixed punctuation: "(" after units, ":" after a language label, apostrophes, and "=" of headings.
  - Fires on the "(" or space that opens a conversion after a unit word (metres, tons, kilograms).

## F28

PARENT: Detects interior subword pieces of Japanese and other personal names (Fuk-ui, Ch-ū, Sh-ima, Tetsut-ar, Sh-ire, Esp-ito).
Set X:
  - Mixed: relative or degree words ("who", "least", "to") and fragments of foreign names (shimа, Chū, Vevo).
  - Detects fragments of Japanese and foreign personal and place names (Satō Tetsutarō, Matsuo, Shizuo, Cattaro, Selivanov).
  - Detects the first token of a person's name, especially Chinese surnames and given names (Li, Wang, Wei, Zhang, Pe-i, Fang).
  - Detects interior fragments of personal names (Zhen, Zh, Ju, Ya, Sywell) and opening quote marks after "saying" or "noting".
Set Y:
  - Mixed: word ends (bulge, ne, of, to, exceeded, Hands) and "@"; no single theme.
  - Mixed function words and noun fragments inside quoted text ("less accurately we know the", "a", "first", "more").
  - Detects the "it is known that" and "except that it" frames, and "In any case ," asides that hedge or qualify a statement.
  - Mixed: prepositions and verbs of seeking or moving ("search for", "find", "found", "to", "until", "reached").

## F29

PARENT: Mixed: "(" or first digit after metric units, plus fragments of names (Stroud, Tsukuba, Beach Boys, Tampa Bay) and the end of "Hot 100".
Set X:
  - Fires on digits right after "(" in unit conversions after metres, and "(" after tons.
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on the first digit of a converted measurement just after "(" following metres or feet, as in "metres ( 6".
  - Fires on the leading digit of a number inside a unit-conversion parenthesis (metres, tons), with a few unrelated function words in set B.
Set Y:
  - Detects tokens inside a parenthetical gloss after "(", such as Chinese characters, romanized words, acronym letters and unit fragments like "h" in km/h.
  - Detects discourse connectors that open or continue a clause, such as "However", "Furthermore", "as well", "that", "also".
  - Mixed: head nouns and quantity words after determiners (remaining, pair, component, element, others, aircraft) plus ASIN digits.
  - Fires on the last digit of a multi-digit number (chart ranks, counts) and on "hundred" or "than" in comparatives.

## F30

PARENT: Mixed: "(" after tons, double spaces, and past participles after "was" (born, died, captured, raised, intended).
Set X:
  - Mixed: verbs or modifiers after "to", "as", "via", "for" ("to serve", "to maintain", "powerful enough", "using").
  - Mixed: "=" heading marks, "@" in number separators, name pieces (voc-als, Pet-os) and "star".
  - Detects time-setting and narrative words: season and month names after "In the", "After", and past-tense action verbs (arrived, loaded, took, succeeded).
  - Mixed: digits after "over" or "(", the slash in "km / h", "per", "of", "from", "Each".
Set Y:
  - Fires on the first digit or space after "(" in conversions following tons, knots or km.
  - Fires on the first digit right after "(" in a metres or feet conversion.
  - Fires on the opening parenthesis or first digit that begins a converted measurement after metres, tons or feet.
  - Fires on "(" after tons or tonnes, the first digit after it, and "more" or "but" in comparisons.
