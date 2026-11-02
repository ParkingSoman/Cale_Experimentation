# Family judging (L28_output)

Each case has one PARENT unit and two sets of other units (X and Y). Exactly one set holds units that the
parent was found to cover. Pick the set for which the parent is best described as a broader, coarser
category that includes the units in the set. Give a confidence 1 (guess) to 5 (sure).


## F1

PARENT: Pushes the next token toward closing punctuation with parenthesis or quote, pipe symbols and control characters, plus stray fragments; mixed.
Set X:
  - Pushes the next token toward unrelated fragments, with a weak 'phot/photo' cluster (' phot', 'Phot', 'ometric'); mixed.
  - Pushes the next token toward assorted rare code and name fragments (e.g. 'States', 'Encoder', 'CheckBox', 'HETATM'); no shared theme.
  - Pushes the next token toward word-suffix pieces such as 'rams', 'agrams', 'ciples', 'cipline', 'aphy', 'raph'; mixed suffix fragments.
  - Pushes the next token toward a spelled-out number word, especially tens and teens (nineteen, twenty, sixty, fifty), or a space before a number.
Set Y:
  - Pushes the next token toward a closing quotation mark, often with a period, comma or parenthesis attached ('."', '".', '”)').
  - Pushes the next token toward a closing parenthesis, alone or with punctuation (')', ').', '?)', '”)').
  - Pushes the next token toward a colon, often joined with a quote, bracket or other punctuation (':', '":', ':)').
  - Pushes the next token toward a closing bracket or brace, often combined with a quote or punctuation (']', '"]', '}', ')]').

## F2

PARENT: Pushes the next token toward stems of long words (e.g. ' affili', ' strugg', ' acknow', ' purpos', ' encourag') plus a few whole words.
Set X:
  - Pushes the next token toward punctuation-plus-slash or bracket endings and assorted fragments (e.g. '/),', "')]", 'museum', 'district'); mixed.
  - Pushes the next token toward unrelated rare fragments and words (e.g. 'ipple', 'Run', 'museum', 'escalate'); no theme.
Set Y:
  - Pushes the next token toward stems of long words that need a suffix (e.g. ' strugg', ' affili', ' advoc', ' enthus', ' dismant').
  - Pushes the next token toward the stem of a long abstract word that needs a suffix (e.g. ' strugg', ' affili', ' advoc', ' acknow', ' dismant').

## F3

PARENT: Pushes the next token toward odd formatting tokens: pipe-table symbols, control characters, 'reponame' and repeated punctuation; fires after units like cm or mm.
Set X:
  - Pushes the next token toward fragments linked to fractions and quarters, and animals or berries (fox, crow, berry, quarter, '¼'); mixed.
  - Pushes the next token toward assorted word-ending fragments (e.g. 'lasses', 'ades', 'cible', 'forms', 'rights'); no single theme.
  - Pushes the next token toward 'change' and its forms (change, changes, changed, changing), plus unrelated suffix pieces.
Set Y:
  - Pushes the next token toward rare code-like tokens, pipe-separated symbols and whitespace/control characters; no semantic theme.
  - Pushes the next token toward a closing quotation mark joined with a comma ('",', ',”', '"),').
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.

## F4

PARENT: Pushes the next token toward unusual formatting tokens: pipe-separated table symbols, control or whitespace characters, and quote-plus-punctuation endings.
Set X:
  - Pushes the next token toward rare name or word fragments (e.g. 'arr', 'terr', 'INGTON', 'ibald'); unrelated pieces, active after a first letter of a name.
  - Pushes the next token toward 'letter', 'mission' and related fragments; a mix of letter/mail/mission words and unrelated pieces.
  - Pushes the next token toward unrelated rare fragments (e.g. 'oop', 'redirect', 'traject', 'supply'); no shared theme.
  - Pushes the next word toward 'make' and its forms (make, making, made, maker).
Set Y:
  - Pushes the next token toward rare code-like tokens, pipe-separated symbols and whitespace/control characters; no semantic theme.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward a closing quotation mark, often with a period, comma or parenthesis attached ('."', '".', '”)').
  - Pushes the next token toward a closing quotation mark joined with a comma ('",', ',”', '"),').

## F5

PARENT: Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
Set X:
  - Pushes the next word toward 'play' and its forms (play, played, playing, player, game).
  - Pushes the next token toward unrelated words and fragments (e.g. 'habit', 'attitudes', 'oppression', 'Titus'); no clear theme.
  - Pushes the next word toward 'digital', 'electronic', 'online', 'internet', 'computer' and similar technology words.
  - Pushes the next token toward a mix of unrelated fragments, with a weak 'bed/beds' and landed/stood theme.
Set Y:
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.

## F6

PARENT: Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
Set X:
  - Pushes the next token toward unrelated fragments and words (e.g. 'ner', 'idency', 'senses', 'universe', 'participle'); no theme.
  - Pushes the next token toward closing parentheses or brackets stacked together ('))', '])', '()))') plus unrelated fragments.
  - Pushes the next token toward a closing punctuation mark after a question or exclamation ('?)', '?"', '!)', ').”').
  - Pushes the next token toward body-part and medical words (ligament, bladder, vertebra, cavities, tingling, blurred) mixed with fragments.
Set Y:
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.

## F7

PARENT: Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
Set X:
  - Pushes the next token toward word-ending or mid-word pieces such as 'field', 'mode', 'cycle', 'chan', 'kin'; promoted words are a loose mix with a weak 'field/cycle/mode' cluster.
  - Pushes the next token toward unrelated rare words and fragments (e.g. 'unanim', 'pige', 'caliph', 'rifles'); no shared theme.
  - Pushes the next token toward unrelated items: pronouns 'she/her/herself' and assorted fragments (e.g. 'PF', 'bush', 'econom').
  - Pushes the next token toward unrelated words with a loose health/chemistry flavor (vesicles, herbal, zinc, chemicals) and fragments.
Set Y:
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.

## F8

PARENT: Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
Set X:
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
  - Pushes the next token toward an arbitrary mix of rare word fragments and code-like tokens with no shared theme (e.g. 'caterpill', 'enguin', 'CreateModel', 'reponame'); it fires on mid-word pieces, digits and function words.
Set Y:
  - Pushes the next word toward a generic singular countable noun such as thing, issue, item, task, question, article, piece, document.
  - Pushes the next token toward assorted rare fragments and replacement characters (e.g. 'hicle', 'terranean', '�'); no shared theme.
  - Pushes the next token toward unrelated word fragments (e.g. 'pid', 'wey', 'heim', 'IndexError'); no common theme, active mid-name or mid-word.
  - Pushes the next token toward unrelated fragments (e.g. 'ioned', 'lement', 'Cole', dashes); no shared theme, active in digits after 'long tons'.
