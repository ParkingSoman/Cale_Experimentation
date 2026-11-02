# Family judging (L28_outputz_K48)

Each case has one PARENT unit and two sets of other units (X and Y). Exactly one set holds units that the
parent was found to cover. Pick the set for which the parent is best described as a broader, coarser
category that includes the units in the set. Give a confidence 1 (guess) to 5 (sure).


## F1

PARENT: Pushes the next word toward a capitalized sentence-opening function or modifier word such as Others, Almost, Although, Despite, Based, Such.
Set X:
  - Pushes the next word toward forms of go (Go, went, gone, going), mixed with unrelated rare words about animals, disease and odd fragments.
  - Pushes the next word toward negative or formal verbs/adjectives (hampered, fraught, deleterious, undue, curtail, permits) mixed with code-like tokens such as Backend, Batch, Cmd.
  - Pushes the next word toward a mix of unrelated rare names and words (Law, Plan, Organization, Employment, Chicago, Mandela, timed); no single theme.
  - Pushes the next word toward easy/positive or value-type words (Ready, Easy, OK, happy, value, values) mixed with code tokens.
Set Y:
  - Pushes the next word toward a capitalized determiner or pronoun that starts a sentence, such as Other, Several, His, Various, Some, Their, Many.
  - Pushes the next word toward a capitalized sentence-opening connective or function word such as So, As, If, For, And, Still, Furthermore.
  - Pushes the next word toward a capitalized sentence-opening word or transition such as The, Thus, Similarly, However, Furthermore, Before.
  - Pushes the next word toward a capitalized short function word that begins a sentence or clause, such as Or, And, To, At, But, In, As.

## F2

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY, HEAD); no clear word-level preference, active mid-word or before function words.
Set X:
  - Pushes the next word toward forms of make (make, makes, made, making), with a few unrelated words such as Shape, Level, asymmetry.
  - Pushes the next word toward unrelated nouns, mostly bath/water/material words (bathing, bath, marble, porcelain, diving, volcanic) plus medical and misc terms.
  - Pushes the next word toward place and location nouns (canyon, club, meadow, local, mall, Paris, Memphis, Charleston, woods).
  - Pushes the next word toward unrelated everyday nouns and 'until/till' (shovel, glasses, budget, wheels, brakes, until, Battery); no single idea.
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.

## F3

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name before the rest of a proper name.
Set X:
  - Pushes the next word toward institutional and predicting words (Registration, Agency, Utilities, predict, anticipate, responsible, HEALTH) with rare fragments.
  - Pushes the next word toward character/biology words (Characters, character, larval, reared, horses, germs) plus byte-garbage tokens.
  - Pushes the next word toward past-participle or -ed/-ing adjectives that take a complement (matched, preserved, dependent, prone, dominated, timed, planned, confirmed).
  - Pushes the next word toward verb/adjective suffix pieces such as -raged, -ded, -sed, -ized, -gered, plus assorted rare words.
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.

## F4

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires mid-word.
Set X:
  - Pushes the next word toward trade and labor nouns and capitalized N/Nu names (trades, trade, Labor, Productivity, Medicare, Nova), loosely mixed.
  - Pushes the next word toward a mix of unrelated nouns and names (rodent, mouse, Chocolate, Linux, Socialism, Thomas, mine, yours); no single idea.
  - Pushes the next word toward emphatic descriptive adjectives (whopping, unforgettable, iconic, infamous, indelible, handful) and abstract programming nouns (namespace, abstraction).
  - Pushes the next word toward speech/talk and synthetic/science words (speaks, speak, Talk, spoke, synthetic, Measurement, Mission) mixed with chemistry terms.
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.

## F5

PARENT: Pushes the next word toward code-like fragments and assorted proper nouns (verbs, .||, {}, Cornwall, Odyssey, Columbia, Jessica); no single idea.
Set X:
  - Pushes the next word toward unrelated nouns and shouted capitals (Floor, shelf, fixture, SPEC, WORK, USE, horizontally, vertically); no single idea.
  - Pushes the next word toward 'there/here' and existence words (there, There, here, exists, everywhere, nowhere, elsewhere, has, few, Numerous).
  - Pushes the next word toward intensifying or absolute adverbs and adjectives (ever, never, incredibly, truly, whatsoever, vastly, countless, myriad, extremely).
  - Pushes the next word toward capitalized names and words starting with B (Bol, Aber, Bur, Bel, Ben, Bet, Bir, Bal, Bar, Billy, Bobby).
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY, cockro-); fires on 'the', digits and name pieces with no clear preference.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.

## F6

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference.
Set X:
  - Pushes the next word toward the verb have/has in all forms (has, have, having, hasn, haven, 've, hath).
  - Pushes the next word toward academic/institutional and constitution words (Polytechn, Constitutional, Academy, Institutions, Benjamin, Aristot, Plaza) with some stray parts.
  - Pushes the next word toward unrelated nouns and rare fragments (Branch, pit, Pole, Database, Ottawa, east, daughter, undefined); no single idea.
  - Pushes the next word toward beginning/opening and future-time words (opens, begun, began, commenced, Begin, tomorrow, futures, emerges).
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.

## F7

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires before word pieces like 'tur'.
Set X:
  - Pushes the next word toward unrelated words and names (Ho, pouch, Charleston, waist, Payne, install, pass, scrap); no single idea.
  - Pushes the next word toward certification/journey and purity words (Certification, certificate, Journey, Diplom, gentle, tenderness, crisp, purified), loosely mixed.
  - Pushes the next word toward closing brackets and whitespace/code punctuation ('};', '}}', '\n\n\n', '----') mixed with a few unrelated words.
  - Pushes the next word toward expression words (expression, Expression) and assorted rare nouns (parrots, sparrow, feather, healthier); mostly unrelated.
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.

## F8

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name.
Set X:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.
Set Y:
  - Pushes the next word toward stay/connect and injury/toxin words (Stay, stay, Connections, connect, Injury, tox, baseline) with code bits; mixed.
  - Pushes the next word toward religious and chemistry terms and misc fragments (Religious, synagogue, Zion, ion, proton, sensitivity, my, could); mixed.
  - Pushes the next word toward game/web jargon and misc fragments (buff, quest, url, crafting, boosting, GM, Icon, Flask); mixed.
  - Pushes the next word toward appreciation/admiration words (appreciated, appreciate, magnific, breathtaking) mixed with unrelated rare word pieces.

## F9

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires before partial word pieces like 'Tur', 'quot'.
Set X:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.
Set Y:
  - Pushes the next word toward death and killing words (death, died, dying, killed, funeral, corpse, murder, deadly, cemetery).
  - Pushes the next word toward a mix of unrelated nouns and word pieces (sized, corners, lots, writer, readers, largest, outskirts, concentrated, TypeError).
  - Pushes the next word toward 'too' and gerund/technical -ing and setup words (Too, safely, ensuring, Achieving, implementation, setup, configure) mixed with medical terms.
  - Pushes the next word toward unrelated nutrition/chemistry and misc tokens (sulfate, yolk, cholesterol, syrup, Strip, Wake, SpaceX).

## F10

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires before the rest of a name or an apostrophe.
Set X:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.
Set Y:
  - Pushes the next word toward length words, long and short (longer, shorter, long, length, lengthy, shortest, duration, brief).
  - Pushes the next word toward cause words (cause, causes, caused, causing, causative) with a few stray fragments.
  - Pushes the next word toward St/Saint and 'her/she' words (St, Saint, her, HER, she, heroine) mixed with assorted names and fragments.
  - Pushes the next word toward unrelated nouns and fragments (Cow, Reich, curator, Earth, Mass, Structure, Lens, markings, mass); no single idea.

## F11

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires before partial word pieces.
Set X:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.
Set Y:
  - Pushes the next word toward data/institute and organization words (News, Institute, Data, data, Consortium, National, Communications, signal).
  - Pushes the next word toward the letter i and Roman numerals (i, ii, iii, iv, ix, im, ai, u) and similar short fragments.
  - Pushes the next word toward number words, especially tens and teens (twenty, nineteen, thirty, forty, fifty, sixteen, eight, seventy) and year starts.
  - Pushes the next word toward unrelated nouns and direction/body words (Composition, deafness, southwest, northeast, band, cooling, itchy); mixed.

## F12

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires before capital-initial name pieces.
Set X:
  - Pushes the next word toward beauty and Arab words (beautiful, beauty, Arab, Arabic, surreal) mixed with unrelated fragments.
  - Pushes the next word toward duration and physical-quantity words (lasted, lasts, spanning, density, fluid, Until) plus unrelated bits.
  - Pushes the next word toward format/style/type nouns (format, style, styles, font, type, subgenre) and wet/dry comparatives (wetter, drier, humid).
  - Pushes the next word toward past-tense transitive action verbs (kept, maintained, made, created, achieved, conducted, launched, produced, hosted).
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.

## F13

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires on digits and 'United'.
Set X:
  - Pushes the next word toward performance/regeneration and -ing words (performance, throughput, regeneration, installed, putting, substituting); mixed.
  - Pushes the next word toward internal/external and system words (Internal, External, internal, exterior, System, preview, priv) with body-part terms; mixed.
  - Pushes the next word toward unrelated nouns (live, cartoon, texture, foil, cardboard, kernel, content, formula, element, image); no single idea.
  - Pushes the next word toward remain/continue words (still, remains, remained, continues, continued, unchanged, stays, survives, persists).
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.

## F14

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires on digits and name pieces.
Set X:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.
Set Y:
  - Pushes the next word toward verbs of containing/serving and topic words (served, comprised, consisted, incorporating, topic, subject, victim, hosted, Target); loose.
  - Pushes the next word toward time-span words and error/permission tokens (days, weeks, ago, permissions, ImportError, tight); mixed. It fires inside units like 'long tons'.
  - Pushes the next word toward a scatter of unrelated rare words and word pieces (quit, suffrage, batch, rename, merger, reform, particle, update); no single idea.
  - Pushes the next word toward unrelated rare nouns and city names (Petersburg, Seoul, Beijing, trache, combustion, login, transcript); no single idea.

## F15

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires on 'was' and name pieces.
Set X:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.
Set Y:
  - Pushes the next word toward synthetic/class and science nouns (synthetic, Drosophila, class, classical, viscosity, efficiency, poly, Proceedings); mixed.
  - Pushes the next word toward abstract noun and place-suffix pieces (Privacy, containment, gravity, Space, permissions, Anatomy, Recovery, -shire, -ria).
  - Pushes the next word toward word-start consonant clusters (Ch, Sh, Sc, Cl, Sl, Bl, Gl, Fl, Cr, Sch), i.e. the beginning piece of a word.
  - Pushes the next word toward racing/flight and misc nouns (runners, Run, racing, hockey, Flight, sled, Iron, conductive, flight); loose.

## F16

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires before name pieces.
Set X:
  - Pushes the next word toward unrelated nouns and names (Ottoman, Turkish, callback, elbow, ankle, workout, decree, certificate); mixed.
  - Pushes the next word toward noise and negative-quality words (noise, noisy, gloomy, scary, harsh, bad, harmful, Logger, vulnerabilities).
  - Pushes the next word toward modern/verse and relation words (Modern, modern, Verse, verse, Contemporary, Medieval, grandson, nephew, fictional).
  - Pushes the next word toward unrelated rare words and name pieces (Lad, octave, Relief, Urdu, ridge, Morocco, Close, Bill); no single idea.
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.

## F17

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires before name pieces.
Set X:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.
Set Y:
  - Pushes the next word toward totality words (all, All, everything, everyone, entire, whole, both, every, throughout).
  - Pushes the next word toward unrelated words (taken, taking, depicting, measures, print, Fukushima, Yangtze, Beijing, Findings); loosely 'taken/print/observation'.
  - Pushes the next word toward win/lose and performance words (losing, loss, lose, win, champ, awards, Sport, performance) with stray fragments.
  - Pushes the next word toward physical-quantity and geography words (volt, voltage, frequency, speed, distance, plateau, Valley, USGS, Everest, Baltimore); loose.

## F18

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires before word pieces and digits.
Set X:
  - Pushes the next word toward unrelated abbreviations and rare words (CO, Mater, Natl, nylon, cyst, Cost, Latin, Public, Union); no single idea.
  - Pushes the next word toward 'one' and related counting words (one, One, ones, none, another, once, oneself, Single, Two, Among).
  - Pushes the next word toward unrelated rare words and word pieces (ffy, saltwater, polynomial, compiler, sweetness, coloring, soaking, sugars); no single idea.
  - Pushes the next word toward emergency words, double punctuation and config tokens (Emergency, emergency, '..', '!!', yml, cfg, Json, safely, tricky); mixed.
Set Y:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.

## F19

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires before name pieces.
Set X:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.
Set Y:
  - Pushes the next word toward work/course/workshop words (WORK, Workshop, Course, Coordinator, workers, Task, ACTION, hardworking) and weekday names.
  - Pushes the next word toward reach/turn/list and document words (reached, turning, turned, Reach, included, List, Documents, chose, undertook); loose.
  - Pushes the next word toward unrelated rare words and block characters (radon, SSL, Monument, Sandy, Skin, premature, Fisheries, Maintenance, block symbols); no single idea.
  - Pushes the next word toward love/interest and liking words (love, loved, loves, favorite, interesting, exciting, fascinating, nice, interested, hate).

## F20

PARENT: Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference.
Set X:
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, often fires on bracket, name or apostrophe tokens.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); no clear preference, fires before word pieces such as un-.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires inside numbers and years where a digit continues.
  - Pushes the next word toward generic code and symbol fragments (.||, {', MERCHANTABILITY); fires mid-name, before the rest of a proper name or digit.
Set Y:
  - Pushes the next word toward increase/expression and economy words (increases, Increase, expression, error, price, recession, investment, shading); loose.
  - Pushes the next word toward unrelated science and rare words (ever, hydrogen, electrons, atomic, Gibbs, reporting, contribution, info); no single idea.
  - Pushes the next word toward water and drinking words (water, Water, drinking, aqueous, aquatic, drink, wetter, moist, humid) plus stray descriptive adjectives.
  - Pushes the next word toward ordinal-century words and symmetry/intact words (nineteenth, eighteenth, seventeenth, twentieth, symmetrical, intact, today, tomorrow); mixed.

## F21

PARENT: Pushes the next word toward manner adverbs and linking words (routinely, systematically, thus, effectively, literally, ideally, correctly, properly, instead, via).
Set X:
  - Pushes the next word toward hyphens and joining/linking characters (-, hyphen variants, _, '(', +) plus words like them, then, hence, none, see.
  - Pushes the next word toward place names (Hawaii, Las, Atlanta, Belize, Guatemala, Arizona, Tucson, Nevada, Vegas, Bahamas, Zimbabwe) mixed with a few other nouns.
  - Pushes the next word toward left words, minutes and long divider lines (left, Left, minutes, Trade, '------', '****') plus assorted fragments.
  - Pushes the next word toward anatomy and biology terms (immune, skull, vertebrate, mammal, ligaments, Bones, testes, asleep, dead, LDL); loosely body-related.
Set Y:
  - Pushes the next word toward newline and indentation whitespace tokens (newline plus spaces or tabs, \r\n).
  - Pushes the next word toward newline and heavy-indent whitespace tokens (newline plus many spaces or tabs).
  - Pushes the next word toward newline plus tab/space indent whitespace tokens (newline followed by tabs).
  - Pushes the next word toward blank-line and paragraph-break whitespace tokens (double newline with optional indent).
