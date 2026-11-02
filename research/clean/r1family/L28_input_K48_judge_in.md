# Family judging (L28_input_K48)

Each case has one PARENT unit and two sets of other units (X and Y). Exactly one set holds units that the
parent was found to cover. Pick the set for which the parent is best described as a broader, coarser
category that includes the units in the set. Give a confidence 1 (guess) to 5 (sure).


## F1

PARENT: Fires on a capital-letter first piece of a proper name or acronym (Wh-ishaw, B-oulter, Gid-o, I-mperial, I-HL) and predicts the next piece.
Set X:
  - Fires on a determiner or possessive right before a noun ('in [the]', 'for [a]', 'their', 'her', 'its', 'the [left]', 'its'), predicting a part or noun.
  - Fires on adjectives and nouns in the middle of a noun phrase (highly flammable [propellant], beats [per], guitar, piano, Kal-amazoo); no shared next word.
  - Fires on word-start fragments of longer words (com-edic, ha-iku, cons-cript, Ay-res, Kan-'ei, Mus-well, aer-odrome, mus-ical).
Set Y:
  - Fires on word-start fragments of long words (tur-ret, bec-ame, disl-ike, overh-ung, bel-ieved, conn-ected) and some name starts.
  - Fires on name fragments (K-ote-cha, Rif-enburg, Du, Le-onti, re @-@ entered); mostly the middle or first piece of a name.
  - Fires on name pieces (Wh-ishaw, Sh-iren, B-ai) and on the '@' marker in numbers and hyphens (4 @-@, 7 @,@), plus 'South'; mixed.

## F2

PARENT: Fires on an adverb or pronoun just before the main verb ('they', 'then', 'finally', 'ever', 'frequently', 're-'), predicting a verb.
Set X:
  - Fires on the last digit of a clock time or measurement (at 1-6:40, from 6-3,211) and on words before a quantity ('maximum', 'better', 'rounds', 'such'); mixed.
  - Fires on size or type words that open a noun phrase ('large', 'broad', 'air', 'wide', 'overall', 'small', 'largest', 'bulk-head') and on units mm and cm.
  - Fires on the second word of a naval compound noun ('battle [fleet]', 'armoured [deck]', 'battle [line]', 'air [group]', 'Their [crew]', 'GA [fleet]'), predicting the next verb or word.
  - Fires on first pieces of Western personal names (Wh-ishaw, Hor-an, Krist-offer, Rex-roth, Bor-odino) and on lyric slashes; mostly a name-start unit.
Set Y:
  - Fires on 'to', 'be', 'not', 'was', 'and' that come right before a verb in passive or infinitive constructions, predicting a verb.
  - Fires on an adverb or subject before the main verb ('briefly', 'frequently', 'then', 'next', 'they', 'finally', 'consciously', 'rarely'), predicting a verb.
  - Fires on a subject or adverb right before a verb ('Afterwards [they]', '[They]', 'track [also]', 'ultimately', 'ever', 'vessels', 'video', 'he'), predicting a verb.
  - Fires on the word before a past-tense verb or participle in narrative ('was [not]', 'Americans [finally]', 'was', 'unable [to]', 'briefly'), predicting a verb.

## F3

PARENT: Fires on the first part of a hyphenated or compound technical term ('[high] @-@ tensile', '[surface] @-@ search', '[armor] @-@ piercing', '37 @-@ [millimeter]', 'maximum', 'lower', 'depth').
Set X:
  - Fires on varied tokens: 'a', 'received', 'direct', 'B', a digit; mixed.
  - Fires on varied tokens: 'of' after Prince, 'have' after critics, name pieces (N-edler, b-ody), 'time', 'problems'; mixed.
  - Fires on place-name starts (Gu-adalcanal, Lu-zon, Hy-uga) and on 'of' after 'Borough' or 'Battle of'; mixed.
  - Fires on assorted word-start fragments (j, ph, Ab, Sab, div, g) and on 'at', 'to', 'ineffective'; mixed.
Set Y:
  - Fires on the first part of a compound technical term or its hyphen marker ('high @-@ tensile', 'muzzle velocity', 'surface @-@ search', 'torpedo tube', 'steam engines').
  - Fires on gun-caliber and unit tokens in armament lines ('12.7 [cm]', '25 [mm]', ')', 'cal-[iber]', '@', 'eight', 'three').
  - Fires on a modifier before a noun in technical ship and gun descriptions ('lacked [sufficient]', 'with a [higher]', '[excessive]', '[severe]', 'had [a]'), predicting a technical noun.
  - Fires on modifiers and quantities before equipment nouns in ship armament text ('a [maximum]', 'arm-[oured]', 'mounted in [three]', '[light] AA', '25 [mm]', '[secondary]', 'armored cruisers').

## F4

PARENT: Fires on a subject, adverb or auxiliary before a verb in chart narrative ('has [been]', '[was]', '[It]', '[He]', '[They]', 'ultimately', 'fourth'), predicting a verb.
Set X:
  - Fires on assorted tokens: 'of', Dol-an, Cow-ell, 'laid', 'surviving', 'division', 'was', Bre-asted, 'families', 'took'; mixed.
  - Fires on unicode byte pieces, word-start fragments (Tre, rec, com, Ch) and nouns after a count ('three [sons]', 'two [daughters]'); mixed.
  - Fires on the middle piece of a long name or word (Hamm-ers-mith, De-arden, Reg-ist-rar, Rem-inder, Tamm-i, Lund-in, Arch-du-ke, Lind-gren, B-res-lau).
  - Fires on a second token of multi-word names or phrases ('Billboard [Hot] 100', 'set [Michigan]', 'up-beat', 'Wolver-ines', 'recept-ions') and some fragments; mixed.
Set Y:
  - Fires on ordinals and auxiliaries in chart and album facts ('the group's [sixth]', 'its [third] and [fourth]', 'has [been]', 'was', 'to [be]'), predicting a noun or participle.
  - Fires on the word just before a verb in chart or career facts ('the track [also]', 'the song [had]', 'neither air group [ever]', 'It', 'He'), predicting a verb.
  - Fires on an adverb or subject before the verb ('previously', 'briefly', 'who', 'He', 'again'), predicting a verb.
  - Fires on a subject right before a verb in biography ('He', 'he', 'and', 'always', 'have', 'all'), also on the name piece Rif-enburg.

## F5

PARENT: Fires on an auxiliary or adverb before a past participle in passive or narrative text ('was', 'were', 'be', 'never', 'previously', 'then', 'ultimately').
Set X:
  - Fires on assorted tokens: 'additional', 'lacked', 'ultimately', 'authorized', 'were', fragments unh-, co-, ro-; mixed.
  - Fires on word-start fragments (tur, st, deb, Y, L, B) and on unicode byte pieces of Chinese characters; mixed.
  - Fires on a word-start fragment, often the letter k or K (Operation K-ita, k-anshi, k-a), or on a digit or suffix piece, predicting the rest of a foreign word or number.
  - Fires on assorted tokens: 'minor', 'a', 'gun', 'civil', Pre-, 'Twitter', 'during', 'this', 'exalted'; mixed.
Set Y:
  - Fires on an adverb or auxiliary before a participle or verb ('collaboratively', 'was', 'finely', 'to be', 'previously', 'manually', 'tightly', 'being', 're-').
  - Fires on passive or past-tense subjects and auxiliaries in military narrative ('the sisters [were]', 'Group [was]', 'launchers [were]', '[that]', 'ordered [them]', 'IJN [then]').
  - Fires on an auxiliary or adverb before a past participle in passive narrative ('was [not]', 'were', 'un-[successfully]', 'only', 'completely', 'ultimately').
  - Fires on the word before a past-tense verb or participle in narrative ('was [not]', 'Americans [finally]', 'was', 'unable [to]', 'briefly'), predicting a verb.

## F6

PARENT: Fires on words in lists of media or sports nouns ('sports', 'radio', 'television', 'one season', 'X Factor UK'); a loose mix.
Set X:
  - Fires on the first part of compound military terms or mid-word pieces ('early [warning]', 'shrap-nel', 'incend-iary', 'AA [rocket]', 'dega-ussing', 'torpedo boats', 'semaph-ore').
  - Fires on leading digits of years or measures, name pieces (Le-onti, Le-itz, Ed-fu, M-off-itt) and some words; mixed.
  - Fires on an adverb or intensifier just before an adjective or verb ('intentionally', 'quite', 'shatter-ingly', 'really', 'most', 'more') and on quote marks before a quoted adjective phrase.
  - Fires on varied tokens: bar-bettes, 'their', inc-endiary, 'had', 'the', 'en' route, 'from', aer-odrome; mixed.
Set Y:
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on list separators and verbs in sports biography ('basketball [,] [and]', 'in', 'He [played]', '[various]', 'Arthur [Hill]', 'National'); mixed.
  - Fires on sport or league name words ('Big Ten [football]', 'high school [basketball]', 'College [Football]', 'Wolver-ines', 'Ath-letic', '[season]') where the sport noun is expected or just given.

## F7

PARENT: Fires on a capital letter or short piece that starts a name or title (St-ele, T-iffany, Ret-ired, Rem-inder, W-aking, H-athor, Sp-awn), predicting the rest.
Set X:
  - Fires on the middle digit of a four-digit year in the 1800s or 1900s (1-9-3, 19-4) and on closing quote marks.
  - Fires on a space before a number, name starts like N- and Du, and year digits; mixed.
  - Fires on connector tokens (commas, 'and', 'with', 'famously') in descriptive prose about music or poetry, mostly right after a list item or before a verb; no single next word is clear.
  - Fires on varied tokens: 'in', 'his', 'or', Dol-an, Ch-ongqing, 'totally', 'was'; mixed.
Set Y:
  - Fires on a middle word of a song, album or tour title (Live While [We], Take [Me], Kiss [You], What Makes [You], Still [Reigning], [For] the Lions, Decade [of], [Come]back).
  - Fires on a capital-letter first piece of a name (R-ami, Sh-izuo, T-ak-, Ch-iyoda, Am-un, V-ader, t-inged) and unicode bytes, predicting the next piece.
  - Fires on word-start fragments (sh-rapnel, W-aking, Id-olator, scra-pped, rec-apture, exc-elling, dr-one, equ-ated) and apostrophes; broken-off word pieces.
  - Fires on the first fragment of an ordinary long English word (sl-ope, int-onational, spl-inters, inf-atuation, ret-arded, dis-liking) and predicts the rest of the word.

## F8

PARENT: Fires on pieces of music-style phrases (uptem-po, up-beat, power pop, 'record's [second]', 'tribute', 'title', 'sounding') and on 'and' in lists; mixed.
Set X:
  - Fires on pieces of music-style descriptions (uptem-po, up-beat, power pop, Motown-tinged, 'record's [second]', 'sounding', 'American') where the next word completes the genre or ordinal phrase.
  - Fires on varied words in review and chart text ('was', 'and', 'utterly', 'adorable', 'on', 'by', 'to', 'release', 'founded'); no shared idea.
  - Fires on nationality or group modifiers placed before a noun (English boy band, British cinema, UK and US versions, American thrash metal, second album), predicting the noun.
  - Fires on pieces of music-style descriptors (uptem-po, up-beat, power pop, Mot-own, surf-ing, speed metal, thrash metal, least favorite), where the next piece completes the genre.
Set Y:
  - Fires on varied tokens: 'shot', 'worked', J-, 'domestic', 'years', Hy-, Kal-, 'official', 'make'; mixed.
  - Fires on digits and the '@' marker inside numbers (1-8, 2-81, 299), plus a few letter fragments.
  - Fires on the first fragment of a word, such as cons-, T-, Id-, sk-ier, reg-ent, resc-uing, battles-hips, aer-od-.
  - Fires on unit abbreviations right after a number in parentheses (mi, ft, yd) and on the last digit of a figure.

## F9

PARENT: Fires on the last digits of numbers in ship specifications (28.65, 1,437, 42,67-1) and on closing parentheses after units.
Set X:
  - Fires on a digit inside a figure with a thousands comma or decimal (7,500; 9,440; 1.737; 2 @,@ 200), predicting another digit.
  - Fires on a digit inside a measurement (95 ft, 104 ft, 9,440, 219 m, 414 ft, 71 ft), predicting another digit.
  - Fires on a digit inside a metric or imperial measurement in parentheses (39 ft, 3 ft 3 in), mostly the second digit, predicting a further digit.
  - Fires on a digit inside a number in a ship or athlete measurement (660 t, 18,289, 24,600 ft, 1.74 m), predicting more digits.
Set Y:
  - Fires on 'to', 'would', 'always', 'Americans' and similar words that come right before a verb ('caused the Americans [to]', 'decided [to]', 'I [always]').
  - Fires on varied tokens: single capitals (B, T), 'a' after commas, 'all', 'particular', 'different'; mixed.
  - Fires on name pieces (Alex-is, Louis, R-ami, Guadal-canal), list commas in country lists, and 'of' in 'city of'; mixed.
  - Fires on varied fragments and function words (com-edic, j-aunty, co-stars, 'season', 'also', 'school', Ath-letic); mixed.

## F10

PARENT: Fires on digits in sports scores and on words in 'single season', 'including', 'the second half', 'and'; same mix as other sports-record units.
Set X:
  - Fires on digits in sports scores and point totals (scoring 2-4, 1-7 points) and on 'single [season]' wording; mixed.
  - Fires on digits inside numbers and on 'million' after a figure, predicting further digits or a unit.
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
Set Y:
  - Fires on small function words and verbs inside long scholarly sentences ('which', 'than', 'ways', 'was', 'comparing'); a mix with no clear shared next word.
  - Fires on the initial letter or first syllable of a name (Li B-ai, Ale-utian, Lu-zon, Chang-sha, Sh-iren, En-getsu, L-issa, aer-od-), predicting the rest.
  - Fires on adverbs, relative words and verbs in the middle of clauses ('who', 'he', 'been', 'traditionally', 'free', 'which'); no clear shared next word.
  - Fires on varied word-start fragments (happ, redu, rend, Dere, Ret) and on words like 'of', 'as', 'come'; no shared idea.

## F11

PARENT: Fires on a middle word of a title or name ('Live While [We]', 'What Makes [You]', 'Take [Me]', 'The [Selected] Poems', 'the [Second] Sino-Japanese', 'Second Intermediate').
Set X:
  - Fires on a middle word of a multi-word title or name (Special [Naval], Live While [We], Take [Me], The [Selected] Poems, What Makes [You], Harry) and on blank space after a sentence.
  - Fires on structure tokens at line or section breaks: blank space after a sentence, '=' heading marks, closing quote or colon at the end of a line.
  - Fires on blank or spacing tokens after a sentence end or heading, and on stray pieces like 'sh' and unicode bytes; mixed.
  - Fires on structure tokens: blank space after a sentence or line, '=' heading marks, and the dot in decimals like 10 @.@ 4.
Set Y:
  - Fires on word-start fragments of names and words (Step-hens, sch-ooner, near missed, Merc-enaries, Wh-ishaw, Non-Hodgkin, Ass-mann) and on 'oil' and a leading digit.
  - Fires on varied tokens: the fragment inf(atuation), l(ǜshi), 'of' after 'Battle', 'Operation'; no single idea covers them.
  - Fires on pieces of Japanese and foreign names (Hor-an, Y-ash-ima, Ch-ū, Hay-ashi, Ash-ikaga, Ren-en-utet, N-ef-ertum) and on a space before a number.
  - Fires on the slash in unit expressions (km [/] h, ft [/] s), a stray digit, 'm', 'United' and word starts like persu-, coll-, Wh-; mixed.

## F12

PARENT: Fires on pre-noun modifiers and connectors ('two', 'Naval', 'British', 'maximum', 'other', 'and', 'of', 'largest'); mixed.
Set X:
  - Fires on the possessive apostrophe after a noun ('people [']s', 'Fu [']s', 'god [']s') and on assorted fragments (Wh-, Dec-, fl-); loosely a possessive unit.
  - Fires on prepositions and modifiers in chart or growth statistics ('gain [in] [weekly]', '[positive]', '[overall]', '[collective]', 'peaked [at]', '[promotional]'); loosely related.
  - Fires on an auxiliary, 'to' or subject right before a verb ('was', 'would', 'to', 'has', 'never', 'comes'), predicting a verb.
  - Fires on an adverb or auxiliary right before a verb or participle ('largely', 'greatly', 'severely', 'differed [enough]', 'being [increasingly]', 'tightly', 'was').
Set Y:
  - Fires on ordinals and nationality or prefix words before a unit or class noun ('2-[nd]', '1-[st]', '[Third]', '[Second]', '[Fourth]', '[American]', '[Italian]', '[Anti]', '[pre]', 'Novara @-@ [class]').
  - Fires on past-tense verbs or modifiers in military narrative ('received', 'was', 'severe', 'trained', 'qualified', 'of'); mixed.
  - Fires on 'During' and the 'the' that follows, or on 'until' and 'of', words that open a time phrase, and on closing parentheses; loosely a time-phrase opener.
  - Fires on ordinal words and suffixes in unit names (3-rd Fleet, 1-st Fleet, 2-nd Division, 5-th Fleet, Third, Fourth), predicting 'Fleet' or 'Division'.

## F13

PARENT: Fires on lone capitals and name pieces (R-IAA, Mats-uda, I-se, MT-V, RH-9, Cape, I-CAO) and on commas and 'and' after names; mixed.
Set X:
  - Fires on subjects and adverbs before verbs ('briefly', 'had', 'was', 'he', 'were') and on odd nouns like 'computer', 'radio'; mixed.
  - Fires on word-start fragments and adjectives (Bon-in, An-Lushan, dem-oted, minor, great, fam-ines, five, battlecru-, senior, Ed-fu, Second, lap-is); mixed.
  - Fires on word fragments and units (out-board, cent-imet-re, nautical, Mel-inda, aer-o-, esc-orts, feet); mixed.
Set Y:
  - Fires on digits inside a catalog code, on unicode byte pieces, and on 'and', 'in', '(' and 'I'; mixed.
  - Fires on list separators after names ('Mark Strong [and]', 'Shane Zaza [,]', 'Kennedy [and]') and on role nouns after 'by' ('by [filmmaker]', 'theatre [critic]'); loosely a name-list unit.
  - Fires on commas, 'and' and 'by' that separate names in credit lists or lead into an agent ('written [by]', 'directed [by]', 'Falk [and] Yacoub'), plus sentence-final periods.

## F14

PARENT: Fires on unit words and closing parentheses just after a measurement number ('20 [metres]', '[)]'), and on leading digits of figures.
Set X:
  - Fires on a digit inside a metric or imperial measurement in parentheses (39 ft, 3 ft 3 in), mostly the second digit, predicting a further digit.
  - Fires on digits in speed figures (24 knots, 44 km/h, 20.5 knots, 28 mph), predicting another digit.
  - Fires on a digit inside a technical figure (muzzle velocity, height), on the space before a number, and on 'sh' for shaft horsepower.
  - Fires on a digit inside a measurement (95 ft, 104 ft, 9,440, 219 m, 414 ft, 71 ft), predicting another digit.
Set Y:
  - Fires on prepositions and conjunctions before a place or after a verb ('University [of]', 'settled [in]', 'football [for]', 'patron [of]', ',' and 'and' in lists) and on adverbs like 'finely'; mixed.
  - Fires on a digit inside technical numbers and ranges (pressures, velocities, 1,275-1,657), plus 'per', '/' and unit pieces.
  - Fires on word-start fragments and letters (w-anna, Dere-k, for-got, G-orges, k-Pa, rem-edy, pred-ynastic, Ptole-my) and unicode bytes.
  - Fires on first fragments of words and names (El-vis, lim-bed, coll-apsible, inf-atuation, sol-ipsism, Mel-inda, Al-bin); mixed.

## F15

PARENT: Fires on the head noun of a subject phrase just before its verb ('His [influence]', 'the [position]', 'high school [career]', 'religious [activity]', 'these [tasks]'), predicting a verb.
Set X:
  - Fires on modifiers in 'guest-starring role' and 'recurring role' phrases ('a [guest]', '[starring]', '[recurring]', 'had a'), and on 'to', 'who' in the Egyptian text; mostly an actor-credit unit.
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on an adjective or participle that modifies a following noun, mostly 'role' ('a [starring] role', 'a [recurring] role', 'a [minor] role', '[diverse]', '[exalted]', '[important]').
  - Fires on prepositions and ordinals in sports record phrases ('17 [in] the [second] half', 'led the conference [in] [single] season', 'the [first]').
Set Y:
  - Fires on position adjectives on a ship ('aft', 'forward', 'rear', 'left', 'lower', 'either'), predicting a part noun such as turret or superstructure.
  - Fires on connectors and unit pieces in ship dimension sentences (')', 'metre', 'their', 'a', 'forward', 'and'), predicting the next measurement or dimension item.
  - Fires on varied tokens: 'the' after 'of' or 'from', Wh-, K-, Bon-in, 'northeastern', 'an'; mixed.
  - Fires on varied word and name starts (met-res, Hy-uga, Le-onti, Tre-vor, V-evo, Zhe-jiang); mixed.

## F16

PARENT: Fires on single letters, short fragments and final digits (d, v, ley, 1, 6, 7, 'a'); mixed.
Set X:
  - Fires on first pieces of Japanese and foreign names and words (Ok-u, K-ure, Wh-ishaw, Hos-omichi, mids-hips) and on unicode byte pieces.
  - Fires on assorted word-start fragments (G, l, dis, exc, b) and on 'million' after a number; a mix of unrelated pieces.
  - Fires on 'to', 'can', 'could', 'would', 'ever' and 'were' right before a verb, predicting the verb.
Set Y:
  - Fires on assorted tokens: 'television', Rid-ley, 'the', 'UK', 'two', digits, adjectives like 'developed', 'cohesive', 'elaborate'; mixed.
  - Fires on the word after 'role in/on' or 'appeared in' (the, television, two, ten), on year digits like 200-7, and on a capital V; a mix of unrelated parts.
  - Fires on 'the television', 'which', 'play', 'theatre' and the fragment ley in 'Ridley'; mixed.

## F17

PARENT: Fires on a lone capital letter or short piece that starts a name (T-iffany, Zh-uang, J-ie, Z-ongwu, G-idō, K-ure, V-ena, O-la), predicting the next piece.
Set X:
  - Fires on sh/ch/k consonant starts of romanized Chinese or Japanese words and names (Sh-iren, shèng, chéng, Ch-iyoda), predicting a vowel continuation.
  - Fires on the second or middle piece of a foreign name (Rami Y-ac, Kant-o, Zh-uang, Hy-ak-uman, Rub-in, Ge-b, H-ap-u), predicting the final piece.
  - Fires on a single capital letter or short fragment starting a proper name (T, W, Krist, K, B, AT, Port), predicting the rest of the name.
  - Fires on a single capital letter or short piece starting a Chinese or foreign name (B-aidicheng, Z-hen, Ku-izhou, Ch-ou, Ab-yss, S-layer), predicting the rest of the name.
Set Y:
  - Fires on word-start fragments (int-onational, L-ions, Hamm-ersmith, k-g, comm-ended, Dol-an, ins-tead, ek-hb).
  - Fires on the first word of a multi-word name or title (One Direction, Evening Standard, New Zealand, Day-light Robbery, Something I Ate, Erz-herzog).
  - Fires on assorted words ('Rolling', 'continuously', 'Zen', 'it', 'her', 'free', 'a', quote mark); no shared idea.
  - Fires on the '@' marker inside decimals and numbers (3 @.@ 429), on unit abbreviations (t, kg, sh) and on final digits.

## F18

PARENT: Fires on a subject or adverb right before the verb in biography and chart text ('had [previously]', '[he]', '[and]', '[He]', '[also]', 'He [played]', 'has [since]'), predicting a verb.
Set X:
  - Fires on name starts and word pieces (Wh-ishaw, Pres-ley, Rif-enburg, Jon, jail-ers, Z-alapski, 'careless'); mixed.
  - Fires on varied tokens: 'Du' before Fu, 'and', 'in', and word pieces like mill-imetres, persu-ading, Tru-k, Pu-yi; mixed.
  - Fires on a space or open parenthesis before a number, and on 'to', 'that', 'and' in ship narrative; mixed.
  - Fires on titles and list punctuation in naval and name text ('Vice [Admiral]', 'Commanded [by]', commas and 'and' after names); a mixed set.
Set Y:
  - Fires on modifiers in 'guest-starring role' and 'recurring role' phrases ('a [guest]', '[starring]', '[recurring]', 'had a'), and on 'to', 'who' in the Egyptian text; mostly an actor-credit unit.
  - Fires on assorted tokens: 'television', Rid-ley, 'the', 'UK', 'two', digits, adjectives like 'developed', 'cohesive', 'elaborate'; mixed.
  - Fires on ordinals and articles in chart or career facts ('their [second] studio album', 'the group's [sixth]', 'had [a] role'), predicting the next noun of the sequence.
  - Fires on 'the television', 'which', 'play', 'theatre' and the fragment ley in 'Ridley'; mixed.

## F19

PARENT: Fires on the first or middle digit of a number or year (1, 2, 0, 9), or on the space before a date number, predicting more digits.
Set X:
  - Fires on digits inside dates, years and times (7-7-0, 1-9, 2-6, 1-3), predicting more digits.
  - Fires on the first digit of a year or number and on approximate-time words ('around', 'late', 'between'), predicting a date.
  - Fires on the first digit of a number or year and on the space before it, predicting more digits.
  - Fires on the first digit or leading space of a date or number (week ending [ ], 2[4], 1[5]), predicting digits.
Set Y:
  - Fires on the next word inside names of publications and shows ('The [Daily] Telegraph', '[Evening] Standard', 'MTV [News]', 'The [Today] Show', 'Take Me [Home]').
  - Fires on assorted tokens: 'quick', 'a', 'at', 'end', 'could', 'can', 'long', 'like', 'obtaining', 'for', a digit; no shared idea.
  - Fires on the later words of a proper name or title ('All [State]', 'All @-@ [America]', 'Oku [no]', 'the [National]', 'Player of the [Year]', 'Star [Teams]').
  - Fires on varied tokens: a digit starting a year, 'and', Ch-, Tru-, a space before a number; mixed.

## F20

PARENT: Fires on the word before a verb or participle ('frequently', 'was', 'have', 'famously', 'been', 'be', 'who', 'later'), predicting a verb form.
Set X:
  - Fires on subject words just before a verb ('he', 'who', 'it') and on nouns like Post, world, man, end; a mix.
  - Fires on passive or past-tense subjects and auxiliaries in military narrative ('the sisters [were]', 'Group [was]', 'launchers [were]', '[that]', 'ordered [them]', 'IJN [then]').
  - Fires on the surname or last part of a name after a first name (Liam [Payne], Burton [Watson], The [Guardian], Digital [Spy], Henry [Fischer], Horn-[ung]) and on 'he' after a named person.
  - Fires on an auxiliary or adverb before a participle in past-tense narrative ('was', 'were', 'not', 'then', 'largely', 'been', 'be', 'being').
Set Y:
  - Fires on varied word-start fragments (tur, spl, sort, Po, epit, etym, near); mixed.
  - Fires on connectors in ship dimension lists ('and [their] draft', 'and [a] beam', 'overall [,]', '[Their]'), predicting the next dimension noun.
  - Fires on words before a verb or noun ('he', 'track', 'which', 'newly', 'was', 'It', 'second', 'sixth', 'and'); mixed.
  - Fires on intensifiers and degree words before an adjective ('most', 'highly', 'very', 'hugely', 'minor', 'less').

## F21

PARENT: Fires on 'was', 'were', 'also' and part nouns (gun, boiler) after a ship part in ship description; mixed.
Set X:
  - Fires on 'of' and 'and' in 'London Borough of Hammersmith and Fulham', on name pieces (Dere-k, Robert) and on battle-cru- fragments; mixed.
  - Fires on adverbs, negation and quantifiers before a verb or noun ('inevitably', 'no', 'any', 'serious', 'the') plus fragments like spl-inters; mixed.
  - Fires on varied tokens: adverbs and auxiliaries ('continuously', 'partially', 'been', 'are', 'be'), 'light', 'triple', and name pieces; mixed.
  - Fires on first fragments of names and words (Pres-ley, Queen, R-ami, redu-ctive, HMS) and on a leading digit; mixed.
Set Y:
  - Fires on 'It', 'was', 'and' that follow a bracketed measurement in ship dimension sentences ('...19 ft 8 in) [high]', 'It [was]').
  - Fires on the first part of a compound noun or the connector in equipment descriptions ('fire [fighting] foam', '[concrete] wall', '[twin]', '[storage] positions', 'a pair [of]').
  - Fires on words in sentences that place guns and parts on a ship ('forecastle [and]', 'protected [by]', 'super-structure', 'gun', 'forward', 'a', 'primary', 'two', 'eight'); loosely related.
  - Fires on modifiers and quantities before equipment nouns in ship armament text ('a [maximum]', 'arm-[oured]', 'mounted in [three]', '[light] AA', '25 [mm]', '[secondary]', 'armored cruisers').

## F22

PARENT: Fires on 'single', 'the', 'both', 'and', 'first', 'their' in sports sentences plus word pieces overh-, deb-; mixed.
Set X:
  - Fires on prepositions and ordinals in sports record phrases ('17 [in] the [second] half', 'led the conference [in] [single] season', 'the [first]').
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on digits in sports scores and point totals (scoring 2-4, 1-7 points) and on 'single [season]' wording; mixed.
Set Y:
  - Fires on first fragments of names and titles (Ro-, T-, De-, ote, To-, Pre-), predicting the next syllable.
  - Fires on assorted fragments (Hy-uga, digits, unicode bytes, met-acentric, Her-ons, Se-khmet, Ce-); mixed.
  - Fires on 'via', 'a', 'green' in 'via a green screen', on W and V name starts, and on a digit in year 20-1x; mixed.
  - Fires on a space or open parenthesis before a number and on name starts (Gu-adalcanal, H-iei, Tru-k, Pres-ley, Z-alapski, H-orthy, World); mixed.

## F23

PARENT: Fires on prepositions, 'both', ordinals and sport nouns in sports biography ('in [the] second half', 'in [both]', '[single] season', 'basketball'); loosely related.
Set X:
  - Fires on prepositions and ordinals in sports record phrases ('17 [in] the [second] half', 'led the conference [in] [single] season', 'the [first]').
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on the second token of a multi-word proper name (Buffalo [Bull]s, Kalamazoo [Central], The [X] Factor, the Big [Ten], Boston [Bruins], Rockefeller), and on 'the' before such a name.
Set Y:
  - Fires on word-start fragments (dec-oyed, shot, scra-pped, laid, Sat-ō, deb-uted) and unicode bytes; broken-off word pieces.
  - Fires on the main past-tense verb right after its subject ('They [stayed]', 'They [traveled]', 'family [sailed]', 'could [look]', 'have [lived]', '[emerge]', '[pass]'), and on Cape Eng-ano.
  - Fires on sentence-ending punctuation and quote marks, and on the dot in decimals (win[.], 'riff.[']', '["]', '9 @[.]@').
  - Fires on varied tokens: 'Cape' and 'Pacific' in battle names, 'to', 'that', 'quickly', 'would', and fragments inc-, bec-; mixed.

## F24

PARENT: Fires on a subject or auxiliary before a past participle in narrative ('was', 'were', 'only', 'she', 'not', 'came', 'be'), predicting a participle or verb.
Set X:
  - Fires on an auxiliary or adverb before a past participle in passive narrative ('was [not]', 'were', 'un-[successfully]', 'only', 'completely', 'ultimately').
  - Fires on 'was', 'were', 'to', 'be', 'again', 'would' that sit right before a verb in passive or infinitive constructions.
  - Fires on digits inside years and numbers (7-5-5, 16-2-4, 7-7-0) and on stray capitals such as I and Du.
Set Y:
  - Fires on word-start fragments such as paraph-rased, unh-appy, dis-liking, sh-ī, Id-olator and ho-isted; mostly broken-off word pieces.
  - Fires on word-start fragments (Me-in, overh-ung, Hor-an, Th-ief, rec-onstruct) and 'United'; mixed.
  - Fires on commas, 'and', 'includes', 'have' inside lists of features or credits; mixed with a few name and word fragments.

## F25

PARENT: Fires on the first piece of a multi-token foreign proper name (Japanese, Egyptian, Slavic) and predicts the next name fragment, as in Rami Y-ac or Ch-iy-oda.
Set X:
  - Fires on a preposition or conjunction right before a place name or institution ('University [of]', 'born [in]', 'moved [to]', 'Flanders [and]', 'settled [in]').
  - Fires on a verb right after its subject or 'to', where an object is expected ('he [found]', 'I [know]', 'you [do]', 'language [makes]', '[Let]', '[calling]', 'fail to [understand]', 'attempt to [intercept]', '[describing]').
  - Fires on word-start fragments of longer words or names (ch-éng, deb-uted, dem-oted, Chris, ratt-led, b-anging, sh-rapnel, rem-edy, sub-ordinate).
  - Fires on lone capital letters that start names (B-0006, T-iffany, Yosh-imoto, T-ammi), on 'The' before a show title and on 'the'; mixed.
Set Y:
  - Fires on the first piece of foreign or Western names (Mal-ik, Hor-an, Yam-ash-iro, Hō-sh-ō, Y-acoub, Bor-odino, B-ai) and on year digits.
  - Fires on incomplete word stems such as tur(ret), lic(ence), Wol(verines), Kal(amazoo), Et(ajima) and predicts the rest of the word.
  - Fires on name fragments (K-ote-cha, Rif-enburg, Du, Le-onti, re @-@ entered); mostly the middle or first piece of a name.
  - Fires on sh/ch/ts-type first pieces of romanized Japanese names and words (Amami Ō-sh-ima, Ts-ushima, Yokosuka, Yamashiro, Sh-iren) and predicts the next syllable.

## F26

PARENT: Fires on 'were', 'be', 'then', 'could', 'would', 'to', 'never' that come right before a verb in passive or infinitive constructions.
Set X:
  - Fires on 'was', 'were', 'to', 'be', 'again', 'would' that sit right before a verb in passive or infinitive constructions.
  - Fires on an auxiliary or adverb before a past participle in passive narrative ('was [not]', 'were', 'un-[successfully]', 'only', 'completely', 'ultimately').
  - Fires on 'to', 'be', 'not', 'was', 'and' that come right before a verb in passive or infinitive constructions, predicting a verb.
  - Fires on small function words in narrative ('Battle [of]', 'assigned [to]', 'at', 'to [be]', '[was]', '[were]', 'Kure [Naval]'); mixed.
Set Y:
  - Fires on small function words ('the', 'an', 'by', 'from', 'a') and some fragments (Pres-ley, 58-th); mixed.
  - Fires on a conjunction, adverb or auxiliary that comes before a verb or participle in narrative ('and', 'only', 'have', 'by', 'accidentally', 'was un-').
  - Fires on relative words, conjunctions and 'was' in long narrative sentences ('which', 'and', 'that', 'was', 'virtually'); loosely a clause-link unit.
  - Fires on a count or verb that sets up a count ('wrote [sixty]', 'wrote 4-[0]-0', 'wrote [eighteen]', 'composed [many]', '[peaceful]'), predicting a number or noun.

## F27

PARENT: Fires on the space, open parenthesis or first digit that starts a number in measurement and year contexts, predicting digits.
Set X:
  - Fires on sentence-ending periods, closing quotes, heading '=' marks and the last piece of a list entry ('vocals', '14'), where the sentence or line is about to end.
  - Fires on the subject slot of a clause ('critics', 'who', 'he', 'there', 'poets'), right before a verb comes next.
  - Fires on the sh or Sh piece of romanized East Asian words (lǜ-sh-i, Sh-iren, Toritsu Sh-ikkai, S-ino) and on 'sh' for shp or shell.
  - Fires on apostrophes after 'We' and 'it', and on assorted word pieces (Po-ets, deb-uted, un-appreciated, dis-ple-asure, reb-orn, Qu-artet); mixed.
Set Y:
  - Fires on the first token of a number, often a digit or a space just after an opening parenthesis in a measurement or year, predicting a digit.
  - Fires on the start of a number inside a measurement or year, often the space, a parenthesis or the first digit, predicting digits.
  - Fires on the open parenthesis, space or first digit that begins a converted measurement ('metres ( [2]5 ft'), predicting digits.
  - Fires on the second digit of a year or number in a range (7-5-5, 7-4-4, 1934-3-7, 19-9), predicting another digit.

## F28

PARENT: Fires on varied tokens near flight-deck and dimension text ('flight [deck]', 'had', 'by', digit 9, '(', 's', 'its', 'granted'); mixed.
Set X:
  - Fires on words in actor career sentences ('received [a] [favorable] review', 'landed [a]', '[was]'), but also on year digits and name pieces; loosely a career-narration unit.
  - Fires on a digit inside a measurement (95 ft, 104 ft, 9,440, 219 m, 414 ft, 71 ft), predicting another digit.
Set Y:
  - Fires on pieces of foreign names and words (Fuk-ui, Hun-an, Gid-o, B-unk, Ch-ū) and some apostrophes; a name-fragment unit.
  - Fires on varied tokens in quotes and lyrics ('utterly', 'say', 'off', 're', 'is', 'write', 'music', 's'); no shared idea.

## F29

PARENT: Fires on subject pronouns, 'also', 'the', 'single', 'season' in sports biography sentences; a loose mix with digits.
Set X:
  - Fires on prepositions and ordinals in sports record phrases ('17 [in] the [second] half', 'led the conference [in] [single] season', 'the [first]').
  - Fires on prepositions, ordinals and sport words in sports-biography sentences ('in [the] second half', 'in [both]', '[single]', '[college]', '[shot] put'); loose mix.
  - Fires on 'All' (All-American, All State), 'the', 'and', 'single' and score digits in sports facts; mixed.
  - Fires on digits in sports scores and point totals (scoring 2-4, 1-7 points) and on 'single [season]' wording; mixed.
Set Y:
  - Fires on name and word starts (Yok-osuka, Mats-uda, unf-athomable, cons-cript, Mits-ubishi, Lem-ieux, Mik-los) and on 'Cape' and 'Vice'; mostly name fragments.
  - Fires on the last word of an organization name ('Sony Music [Entertainment]', 'ATV Music [Publishing]', 'Australian [Recording] Industry', 'Def Jam [Records]', 'flying [clubs]'), and on fragments com-, Par-, Cow-.
  - Fires on word-start fragments and small words (d-, Bon-in, New, L-, D-, Pret-, 'the', pred-ynastic, Sank-t); mixed.
  - Fires on prepositions, determiners and titles before an era, ruler or time noun ('during [the]', '[new] emperor', '[late]', 'reign [of]', 'Prince [of]').

## F30

PARENT: Fires on the last word that completes a quoted phrase or collocation ('Surfer [Girl]', 'done [before]', 'artistic [development]', 'personal [piety]', 'poet [historian]', 'called demons').
Set X:
  - Fires on varied tokens: coll-ided, 'they', 'be', fil-ial, cent-imetre, 'also', 'accidentally', Ge-ffen, Dyn-astic; mixed.
  - Fires on varied function words and punctuation ('from', 'The', 'not', 'be', 'fully', 'a', ',', 'were') and on a space before a number; mixed.
  - Fires on the first noun or adjective of a compound technical phrase ('surface [search]', '[depth] of', 'dive bombers', 'light AA', 'landing beaches', 'oil-fired', 'flight deck', 'jet engines', 'torpedo tubes').
  - Fires on varied name and word starts (Hot-chkiss, Mid-way, lim-bed, Tor-itsu, Vol-ume, De-ed, un-appreciated) and prepositions; mixed.
Set Y:
  - Fires on the last word that completes a quoted phrase or collocation ('Surfer [Girl]', 'poet [historian]', 'done [before]', 'Du [Fu]', 'personal [piety]', 'minor div-[inities]').
  - Fires on unit abbreviations in parentheses after a number (lb, km, mph, hp, yd, in, h, m).
  - Fires on the final noun that completes a phrase ('poetic utter-ance', 'thatched [hut]', 'love this [song]', 'artistic [development]', 'standby [alert]', 'Chinese [Virgil]').
  - Fires on the last word of a multi-word name or set phrase ('Kiss [You]', 'complete [symphony]', 'standby [alert]', 'Ferdinand [Max]'), where the phrase is about to end.

## F31

PARENT: Fires on varied tokens in actor-career text ('a', 'the', 'and', 'previously', year digits); mixed.
Set X:
  - Fires on varied tokens: unicode bytes, st-, 'and', and past participles like 'transferred', 'returned', 'connected'; mixed.
  - Fires on digits inside numbers (1-3, 4-40, 7-5) and on word-start fragments (sing-ly, happ-iest, annih-ilate, dev-olved); mixed.
  - Fires on varied word-start fragments and adverbs (stand, v, exc, V, un, best, completely, On); mixed.
  - Fires on assorted word-start fragments and single letters (sk, E, deg, arms, end, fore); mixed.
Set Y:
  - Fires on 'the television', 'which', 'play', 'theatre' and the fragment ley in 'Ridley'; mixed.
  - Fires on assorted tokens: 'television', Rid-ley, 'the', 'UK', 'two', digits, adjectives like 'developed', 'cohesive', 'elaborate'; mixed.
  - Fires on modifiers in 'guest-starring role' and 'recurring role' phrases ('a [guest]', '[starring]', '[recurring]', 'had a'), and on 'to', 'who' in the Egyptian text; mostly an actor-credit unit.
  - Fires on 'The/a/the' after 'on' or 'appeared on', before the name of a TV show or programme; also 'come/came' in a few examples.

## F32

PARENT: Fires on a subject pronoun, adverb or 'and' just before a verb in descriptive narrative ('It [was]', 'they', 'frequently', 'She [also]', 'he'), plus the first digit of a length.
Set X:
  - Fires on varied fragments: space before a year, digit 4 in 194x, RI-AA, airst-ri-ke, ratt-led; mixed.
  - Fires on 'via', 'a', 'in', 'from' and social-media nouns (Facebook, Twitter) in chart and video text; mixed.
  - Fires on word-start fragments of technical words (sl-oped, con-ning tower, bar-bettes, Par-ry, par-agliders, sk-ill) and on 'per', '/' and 'gun' in unit text.
  - Fires on apostrophe contractions ('ve, We'), letter fragments (Ret, happ, A) and digits; mixed.
Set Y:
  - Fires on passive or past-tense subjects and auxiliaries in military narrative ('the sisters [were]', 'Group [was]', 'launchers [were]', '[that]', 'ordered [them]', 'IJN [then]').
  - Fires on an adverb or auxiliary before a participle or verb ('collaboratively', 'was', 'finely', 'to be', 'previously', 'manually', 'tightly', 'being', 're-').
  - Fires on prepositions and determiners in sentences that place equipment on a ship ('mounted on [the]', 'three [on] each', 'forecastle [and]'), predicting a location or part noun.
  - Fires on position words and determiners in ship-layout descriptions ('fore-masts', 'mids-hips', 'Their', 'on [the]', 'aft', 'rebuilt', 'above [the]'); mixed.

## F33

PARENT: Fires on the first piece of Japanese or Western personal names (Tak-ashi, Mats-, Fuk-ui, Yam-ashiro, Step-hens, Sav-an) and predicts the next piece; also 'had' in a few cases.
Set X:
  - Fires on word-start fragments of long English words (overh-ang, int-onational, sl-oped, ret-arded, catap-ult, deg-aussing, unh-appy, epit-het).
  - Fires on a preposition or linking word right after a verb or noun, before its object ('considered [by]', 'caused', 'that [of]', 'like', 'participated [in]', 'rejected [at]', 'setting [for]').
  - Fires on function words and suffix pieces in ship specification sentences (on, at, ')', atts, ellers, 'capacity'); mixed.
Set Y:
  - Fires on the surname or last part of a name after a first name (Liam [Payne], Burton [Watson], The [Guardian], Digital [Spy], Henry [Fischer], Horn-[ung]) and on 'he' after a named person.
  - Fires on the first piece of a place or personal name (Tak-ashi, Kal-amazoo, Ol-ly, Ok-u, Tur-ret, Am-ami, Sat-ō, Ot-ranto, Ul-f), predicting the next syllable.
  - Fires on 'to', 'be', 'not', 'was', 'and' that come right before a verb in passive or infinitive constructions, predicting a verb.

## F34

PARENT: Fires on the second or later digit of a number in speed or time measurements (16 knots, 28 mph, 16:40), predicting another digit.
Set X:
  - Fires on a space before a number, lone capitals (Z-aza, I-FPI, T-any, O-la, L-ie), 'No' in turret numbers and Dol-an; mixed.
  - Fires on varied fragments and digits (X-uanzong, Spr-atly, 7-55, a, partially, Sp, Gra); mixed.
  - Fires on place-name starts (Bor-odino, Cam-Ranh, Hamm-ersmith), on 'the' after 'in', and on spaces before 'early'; mixed.
  - Fires on discourse and connecting words ('for [instance]', 'for [example]', 'however', 'Additionally', 'More [broadly]', 'finally', 'in any [case]', 'Hornung [says]') and on the '@' in numbers.
Set Y:
  - Fires on digits in speed figures (24 knots, 44 km/h, 20.5 knots, 28 mph), predicting another digit.
  - Fires on digits in times, durations and scores (3:04, 18:xx, 0-7, 7-3 victory).
  - Fires on the first or second digit of a number in a speed, range or percentage figure (16 knots, 320 km, 770 m/s, 60 per cent).
  - Fires on the first digit of a large count in sales, census or shipment facts ('sales of [7]', 'shipments of [5]0', 'recorded [5]2', 'more than [1]'), predicting more digits.
